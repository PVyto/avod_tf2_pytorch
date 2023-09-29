import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from threading import Thread
from utils.box_utils_tf import project_box_to_img, convert_anchors_to_box_format


def predictions_to_kitti_tf(predictions, sample_name, calib_dict, image_shape, experiment_name, current_step=0,
                            mode='val', threshold=0.1, valid_ds=None, save_rpn_predictions=False,
                            calculate_only_proposals=False):
    # print(sample_name)
    if tf.is_tensor(image_shape):
        image_shape = image_shape.numpy().reshape(3)
    if tf.is_tensor(calib_dict):
        new_calib = {}
        new_calib['frame_calib_matrix'] = [None, None, None]
        new_calib['frame_calib_matrix'][2] = calib_dict.numpy()
    if isinstance(calib_dict, list) or isinstance(calib_dict, tuple):
        new_calib = calib_dict[0]

    if isinstance(sample_name, list) or isinstance(sample_name, tuple):
        sample_name = sample_name[0]

    if tf.is_tensor(sample_name):
        sample_name = bytes.decode(sample_name.numpy())
    root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent.as_posix()
    result_folder = root_dir + '/results_tf/' + experiment_name + '/' + mode
    rpn_predictions_path = result_folder + '/rpn'
    avod_predictions_path = result_folder + '/avod'
    kitti_predictions_path = result_folder + '/kitti'

    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(avod_predictions_path, exist_ok=True)
    os.makedirs(kitti_predictions_path, exist_ok=True)

    score = threshold
    data_folder = 'data'
    chpt_folder = 'data_' + str(current_step)

    rpn_chpt = os.path.join(rpn_predictions_path, chpt_folder, sample_name + '.txt').replace(" ", "")
    rpn_predictions = os.path.join(rpn_predictions_path, sample_name + '.txt')

    avod_chpt = os.path.join(avod_predictions_path, chpt_folder, sample_name + '.txt').replace(" ", "")
    avod_predictions = os.path.join(avod_predictions_path, sample_name + '.txt')

    kitti_predictions = os.path.join(kitti_predictions_path, data_folder, sample_name + '.txt')
    kitti_predictions_chpt = os.path.join(kitti_predictions_path, chpt_folder, sample_name + '.txt').replace(" ", "")

    # os.makedirs(kitti_predictions_chpt, exist_ok=True)
    os.makedirs(os.path.join(rpn_predictions_path, chpt_folder), exist_ok=True)
    os.makedirs(os.path.join(avod_predictions_path, chpt_folder), exist_ok=True)
    os.makedirs(os.path.join(kitti_predictions_path, data_folder), exist_ok=True)
    os.makedirs(os.path.join(kitti_predictions_path, chpt_folder), exist_ok=True)

    predictions_rpn, predictions_avod = predictions
    # for RPN get top proposals and their scores for measuring mAP_moderate
    top_anchors, top_scores = predictions_rpn
    # print('rpn: ', len(top_anchors))
    # tensor compatible
    top_proposals = convert_anchors_to_box_format(top_anchors)
    # noinspection PyArgumentList
    proposals = tf.concat([top_proposals, tf.nn.softmax(top_scores, axis=1)[:, None, 1]], axis=1).numpy()
    if save_rpn_predictions:
        pd.DataFrame(proposals).to_csv(rpn_predictions, header=False, index=False)
        pd.DataFrame(proposals).to_csv(rpn_chpt, header=False, index=False)
        if calculate_only_proposals:
            return
    # np.save(rpn_predictions, proposals)

    # use predicted offsets and proposals to calculate the predicted 3d box
    # determine the orientation
    top_scores, top_boxes, top_orientations = predictions_avod
    # print('final: ', len(top_scores))
    # print('top_boxes',top_boxes)
    predictions_3d = top_boxes[-1].numpy().copy()
    top_scores = top_scores.numpy()
    # predictions__ = predictions_3d.copy()
    angle_diff = predictions_3d[:, 6] - top_orientations.numpy()
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi

    ids = (np.pi / 4 < angle_diff) & (angle_diff < np.pi * 3 / 4)
    # print(ids)
    # predictions_3d[ids] = predictions_3d[ids]
    # predictions_3d[ids, [3, 4]] = predictions_3d[ids, [4, 3]]
    predictions_3d[ids, 3:5] = predictions_3d[ids, 4:2:-1]
    predictions_3d[ids, 6] += np.pi / 2

    ids = (-np.pi / 4 > angle_diff) & (angle_diff > -np.pi * 3 / 4)
    # print(ids)

    # predictions_3d[ids, [3, 4]] = predictions_3d[ids, [4, 3]]
    predictions_3d[ids, 3:5] = predictions_3d[ids, 4:2:-1]
    predictions_3d[ids, 6] -= np.pi / 2

    ids = np.abs(angle_diff) >= np.pi * 3 / 4
    # print(ids)

    predictions_3d[ids, 6] += np.pi

    ids = predictions_3d[:, 6] > np.pi
    # print(ids)

    predictions_3d[ids, 6] -= 2 * np.pi
    # get the predicted class
    pred_class = np.argmax(top_scores[:, 1:], axis=1)
    # print('pred class:', top_scores[:, :])

    final_prediction_array = np.zeros((len(predictions_3d), predictions_3d.shape[-1] + 2))

    max_scores = np.array([])
    for i in range(len(predictions_3d)):
        scores = top_scores[i, 1:]
        max_score = scores[pred_class[i]]
        max_scores = np.append(max_scores, max_score)
    final_prediction_array[:, :-2] = predictions_3d
    # torch.stack
    final_prediction_array[:, [-2, -1]] = np.stack([max_scores, pred_class], axis=1)
    # else:
    #     final_prediction_array = predictions
    # if self.save_avod_predictions:
    #     pd.DataFrame(final_prediction_array).to_csv(avod_chpt, header=False, index=False)

    # save_predictions_in_kitti_format
    final_predictions = final_prediction_array[final_prediction_array[:, 7] >= score]
    # print("final predictions:", len(final_predictions))
    if len(final_predictions) == 0:
        open(kitti_predictions, 'w').close()
        # print("sample ", sample_name, "has no valid predictions")
        return
    # print(' self._valid_dataset_.current_calib',  self._valid_dataset_.current_calib)
    boxes_img, ids_to_rm = project_box_to_img(final_predictions[:, :7], new_calib, image_shape)
    # print('projected:', len(boxes_img))
    if len(boxes_img) == 0:
        return None
    predictions_kitti_format = np.zeros((len(boxes_img), 16))
    final_predictions = final_predictions[~ids_to_rm]
    predictions_kitti_format[:, 1:3] = -1.0
    predictions_kitti_format[:, 3] = -10.0
    predictions_kitti_format[:, 4:8] = boxes_img[:, :4]
    predictions_kitti_format[:, 8:11] = final_predictions[:, 5:2:-1]
    predictions_kitti_format[:, 11:14] = final_predictions[:, 0:3]
    predictions_kitti_format[:, 14:16] = final_predictions[:, 6:8]

    predictions_kitti_format = np.round(predictions_kitti_format, 3)
    cls = final_predictions[:, 8]
    cls_name = [valid_ds.classes[int(i)] for i in cls]

    final_predictions_kitti = np.hstack([np.array([cls_name]).reshape(-1, 1), predictions_kitti_format[:, 1:]])
    # print(kitti_predictions)

    pd.DataFrame(final_predictions_kitti).to_csv(kitti_predictions, header=False, index=False, sep=" ")
    pd.DataFrame(final_predictions_kitti).to_csv(kitti_predictions_chpt, header=False, index=False, sep=" ")

    # return final_prediction_array
    return final_predictions_kitti, final_prediction_array, proposals


def call_evaluation_script(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                           return_df=True):

    root_path = root_dir

    evaluation_script_path = root_path / 'kitti_native_eval' / 'run_eval.sh'
    current_epoch = step

    current_experiment_result_folder = root_path / results_parent_folder / experiment_name


    result_folder = current_experiment_result_folder / mode

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'


    results_dir = result_folder / kitti_folder

    plot_folder = results_dir / ('plot_' + str(step))

    executable_path = root_path / 'kitti_native_eval' / 'evaluate_object_3d_offline2'

    result_file = results_dir / ('results_' + str(current_epoch) + '.txt')
    command = " ".join(
        [evaluation_script_path.as_posix(), executable_path.as_posix(), annotation_dir.as_posix(), script_folder,
         str(results_dir), str(current_epoch), experiment_name, str(mode), plot_folder.as_posix().replace(" ", "")])
    print(command)
    os.system(command)


def call_evaluation_script_05(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                              return_df=False):
    root_path = root_dir

    evaluation_script_path = root_path / 'kitti_native_eval' / 'run_eval_05_iou.sh'

    current_epoch = step

    current_experiment_result_folder = root_path / results_parent_folder / experiment_name

    result_folder = current_experiment_result_folder / mode

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'

    results_dir = result_folder / kitti_folder

    plot_folder = results_dir / ('plot_05_' + str(step))

    os.makedirs(plot_folder, exist_ok=True)

    executable_path = root_path / 'kitti_native_eval' / 'evaluate_object_3d_offline2_05_iou_new'

    result_file = results_dir / ('results_05_iou_' + str(current_epoch) + '.txt')



    command = " ".join(
        [evaluation_script_path.as_posix(), executable_path.as_posix(), annotation_dir.as_posix(), script_folder,
         str(results_dir), str(current_epoch), experiment_name, str(mode), plot_folder.as_posix().replace(" ", "")])

    print(command)
    os.system(command)



def calculate_map(**kwargs):
    if kwargs.pop('evaluation_script_to_use') == '05':
        return call_evaluation_script_05(**kwargs)
    return call_evaluation_script(**kwargs)


def calculate_map_async(**kwargs):
    p = Thread(target=calculate_map, args=(), kwargs=kwargs)
    p.start()
