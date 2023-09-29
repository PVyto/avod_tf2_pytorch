import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.box_utils_tf import project_box_to_img, convert_anchors_to_box_format


class CheckpointUtils:
    def __init__(self, optimizer, model, checkpoint_dir, checkpoint_prefix):
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir,
                                                  checkpoint_name=checkpoint_prefix,
                                                  max_to_keep=100)

    def save_checkpoint(self):
        self.checkpoint.save(self.checkpoint_prefix)

    def _restore_checkpoint(self, path):
        self.checkpoint.restore(path)

    def load_checkpoint(self, checkpoint_num):
        checkpoints = sorted([i[:-6] for i in glob.glob(self.manager._checkpoint_prefix + '*.index')])
        checkpoint_dict = {i.split('-')[-1]: i for i in checkpoints}
        print('Available checkpoints : ', checkpoint_dict)
        checkpoint_to_restore = checkpoint_dict.get(str(checkpoint_num))
        self._restore_checkpoint(checkpoint_to_restore)


class TrainerUtils:
    def __init__(self, root_dir, results_parent_folder, experiment_name, annotation_dir):
        self.root_dir = root_dir
        self.results_parent_folder = results_parent_folder
        self.experiment_name = experiment_name
        self.annotation_dir = annotation_dir

    def call_evaluation_script(self, step, mode='val'):
        evaluation_script_path = os.path.join(self.root_dir.replace(" ", ""), 'kitti_native_eval/run_eval.sh')
        current_epoch = step
        current_experiment_result_folder = os.path.join(self.root_dir, self.results_parent_folder,
                                                        self.experiment_name)
        result_folder = os.path.join(current_experiment_result_folder, mode)

        kitti_folder = 'kitti'
        script_folder = 'kitti_native_eval'

        results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
        plot_folder = os.path.join(results_dir, 'plot_' + str(step))
        executable_path = os.path.join(self.root_dir.replace(" ", ""), 'kitti_native_eval/evaluate_object_3d_offline2')
        result_file = os.path.join(results_dir, 'results_' + str(current_epoch) + '.txt')
        command = " ".join(
            [evaluation_script_path, executable_path, self.annotation_dir, script_folder, str(results_dir),
             str(current_epoch), self.experiment_name, str(mode),
             plot_folder.replace(" ", "")])
        os.system(command)

    def call_evaluation_script_05_recall(self, step, mode='val', return_df=False):
        evaluation_script_path = os.path.join(self.root_dir.replace(" ", ""), 'kitti_native_eval/run_eval_05_iou.sh')
        current_epoch = step
        current_experiment_result_folder = os.path.join(self.root_dir, self.results_parent_folder, self.experiment_name)
        result_folder = os.path.join(current_experiment_result_folder, mode)

        kitti_folder = 'kitti'
        script_folder = 'kitti_native_eval'

        results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
        plot_folder = os.path.join(results_dir, 'plot_05_' + str(step))
        os.makedirs(plot_folder, exist_ok=True)
        executable_path = os.path.join(self.root_dir.replace(" ", ""),
                                       "kitti_native_eval/evaluate_object_3d_offline2_recall")
        result_file = os.path.join(results_dir, 'results_05_iou_' + str(current_epoch) + '.txt')
        command = " ".join(
            [evaluation_script_path, executable_path, self.annotation_dir, script_folder, str(results_dir),
             str(current_epoch), self.experiment_name, str(mode),
             plot_folder.replace(" ", "")])

        os.system(command)

    def predictions_to_kitti(self, predictions, sample_name, calib_dict, image_shape, current_step=0,
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
        root_dir = os.path.dirname(os.path.realpath(__file__))
        result_folder = root_dir + '/results_tf/' + self.experiment_name + '/' + mode
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
        kitti_predictions_chpt = os.path.join(kitti_predictions_path, chpt_folder, sample_name + '.txt').replace(" ",
                                                                                                                 "")

        # os.makedirs(kitti_predictions_chpt, exist_ok=True)
        os.makedirs(os.path.join(rpn_predictions_path, chpt_folder), exist_ok=True)
        os.makedirs(os.path.join(avod_predictions_path, chpt_folder), exist_ok=True)
        os.makedirs(os.path.join(kitti_predictions_path, data_folder), exist_ok=True)
        os.makedirs(os.path.join(kitti_predictions_path, chpt_folder), exist_ok=True)

        predictions_rpn, predictions_avod = predictions
        # for RPN get top proposals and their scores for measuring recall
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

        # use predicted offsets and proposals to calculate the predicted 3d box
        # determine the orientation
        top_scores, top_boxes, top_orientations = predictions_avod
        predictions_3d = top_boxes[-1].numpy().copy()
        top_scores = top_scores.numpy()
        angle_diff = predictions_3d[:, 6] - top_orientations.numpy()
        angle_diff[angle_diff > np.pi] -= 2 * np.pi
        angle_diff[angle_diff < -np.pi] += 2 * np.pi

        ids = (np.pi / 4 < angle_diff) & (angle_diff < np.pi * 3 / 4)
        predictions_3d[ids, 3:5] = predictions_3d[ids, 4:2:-1]
        predictions_3d[ids, 6] += np.pi / 2

        ids = (-np.pi / 4 > angle_diff) & (angle_diff > -np.pi * 3 / 4)

        predictions_3d[ids, 3:5] = predictions_3d[ids, 4:2:-1]
        predictions_3d[ids, 6] -= np.pi / 2

        ids = np.abs(angle_diff) >= np.pi * 3 / 4

        predictions_3d[ids, 6] += np.pi

        ids = predictions_3d[:, 6] > np.pi

        predictions_3d[ids, 6] -= 2 * np.pi
        pred_class = np.argmax(top_scores[:, 1:], axis=1)

        final_prediction_array = np.zeros((len(predictions_3d), predictions_3d.shape[-1] + 2))

        max_scores = np.array([])
        for i in range(len(predictions_3d)):
            scores = top_scores[i, 1:]
            max_score = scores[pred_class[i]]
            max_scores = np.append(max_scores, max_score)
        final_prediction_array[:, :-2] = predictions_3d
        final_prediction_array[:, [-2, -1]] = np.stack([max_scores, pred_class], axis=1)

        final_predictions = final_prediction_array[final_prediction_array[:, 7] >= score]
        if len(final_predictions) == 0:
            open(kitti_predictions, 'w').close()
            return
        boxes_img, ids_to_rm = project_box_to_img(final_predictions[:, :7], new_calib, image_shape)
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
