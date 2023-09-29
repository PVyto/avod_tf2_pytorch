import glob
import os
import time
import json
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
from torch.utils.data import DataLoader
from data_utils.datasets_tf import CustomDatasetTf, KITTIDatasetTf
from utils.box_utils_tf import project_box_to_img, convert_anchors_to_box_format

from PIL import Image, ImageDraw

pd.set_option('display.max_columns', None)


def call_evaluation_script_recall(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                                  return_df=True):

    evaluation_script_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/run_eval.sh')
    current_epoch = step
    current_experiment_result_folder = os.path.join(root_dir, results_parent_folder, experiment_name)
    result_folder = os.path.join(current_experiment_result_folder, mode)

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'

    # was exp_path +
    results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
    plot_folder = os.path.join(results_dir, 'plot_' + str(step))
    # os.makedirs(plot_folder, exist_ok=True)
    executable_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/evaluate_object_3d_offline2_recall')
    result_file = os.path.join(results_dir, 'results_' + str(current_epoch) + '.txt')
    command = " ".join([evaluation_script_path, executable_path, annotation_dir, script_folder, str(results_dir),
                        str(current_epoch), experiment_name, str(mode),
                        plot_folder.replace(" ", "")])
    print(command)
    os.system(command)
    # print('reading: ', result_file)


def call_evaluation_script(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                           return_df=True):

    evaluation_script_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/run_eval.sh')
    current_epoch = step
    current_experiment_result_folder = os.path.join(root_dir, results_parent_folder, experiment_name)
    result_folder = os.path.join(current_experiment_result_folder, mode)

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'

    results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
    plot_folder = os.path.join(results_dir, 'plot_' + str(step))
    executable_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/evaluate_object_3d_offline2')
    result_file = os.path.join(results_dir, 'results_' + str(current_epoch) + '.txt')
    command = " ".join([evaluation_script_path, executable_path, annotation_dir, script_folder, str(results_dir),
                        str(current_epoch), experiment_name, str(mode),
                        plot_folder.replace(" ", "")])
    print(command)
    os.system(command)



def call_evaluation_script_05(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                              return_df=False):
    evaluation_script_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/run_eval_05_iou.sh')
    current_epoch = step
    current_experiment_result_folder = os.path.join(root_dir, results_parent_folder, experiment_name)
    result_folder = os.path.join(current_experiment_result_folder, mode)

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'

    results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
    plot_folder = os.path.join(results_dir, 'plot_05_' + str(step))
    os.makedirs(plot_folder, exist_ok=True)
    executable_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/evaluate_object_3d_offline_05_iou_new')
    result_file = os.path.join(results_dir, 'results_05_iou_' + str(current_epoch) + '.txt')
    command = " ".join([evaluation_script_path, executable_path, annotation_dir, script_folder, str(results_dir),
                        str(current_epoch), experiment_name, str(mode),
                        plot_folder.replace(" ", "")])

    print()
    os.system(command)
    df = pd.read_csv(result_file)
    df = df.loc[~(df['experiment_name'] == 'experiment_name')].iloc[[-1], :]

    mAP_easy = df.loc[:, 'detection_AP'].to_numpy().astype(np.float).item()
    mAP_moderate = df.loc[:, 'detection_moderate'].to_numpy().astype(np.float).item()
    mAP_hard = df.loc[:, 'detection_hard'].to_numpy().astype(np.float).item()
    if return_df:
        return df
    return mAP_easy, mAP_moderate, mAP_hard


def call_evaluation_script_05_recall(root_dir, results_parent_folder, experiment_name, annotation_dir, step, mode='val',
                                     return_df=False):
    evaluation_script_path = os.path.join(root_dir.replace(" ", ""), 'kitti_native_eval/run_eval_05_iou.sh')
    current_epoch = step
    current_experiment_result_folder = os.path.join(root_dir, results_parent_folder, experiment_name)
    result_folder = os.path.join(current_experiment_result_folder, mode)

    kitti_folder = 'kitti'
    script_folder = 'kitti_native_eval'

    results_dir = os.path.join(result_folder.replace(" ", ""), kitti_folder)
    plot_folder = os.path.join(results_dir, 'plot_05_' + str(step))
    os.makedirs(plot_folder, exist_ok=True)
    executable_path = os.path.join(root_dir.replace(" ", ""), "kitti_native_eval/evaluate_object_3d_offline2_recall")
    result_file = os.path.join(results_dir, 'results_05_iou_' + str(current_epoch) + '.txt')
    command = " ".join([evaluation_script_path, executable_path, annotation_dir, script_folder, str(results_dir),
                        str(current_epoch), experiment_name, str(mode),
                        plot_folder.replace(" ", "")])

    print(command)
    print()
    os.system(command)


def calculate_map(**kwargs):
    return call_evaluation_script(**kwargs)


def calculate_map_async(**kwargs):
    p = Thread(target=calculate_map, args=(), kwargs=kwargs)
    p.start()


def train_step(model, input_sample, optimizer, step, summary_writer=None, clip_gradients=False):
    model.train()
    with tf.GradientTape() as tape:
        *output, loss = model(*input_sample[:-1], enable_path_drop=True, training=True)

    _gradients = tape.gradient(loss, model.trainable_variables)
    if clip_gradients:
        _gradients = [tf.clip_by_norm(g, 1.) if g is not None else None for g in _gradients]
    optimizer.apply_gradients(zip(_gradients, model.trainable_variables))

    if summary_writer:
        with summary_writer.as_default():
            tf.summary.scalar('total_loss', loss, step=step // 1000)


def fit_from_torch_dataloader(model, dataset_loader, optimizer, clip_gradients, starting_epoch, epochs, checkpoint,
                              checkpoint_prefix, evaluation_interval, evaluation_fn=None, evaluation_fn_params=None,
                              summary_writer=None):
    print('{}: Starting the training process'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if evaluation_fn_params is None:
        evaluation_fn_params = {}
    N = len(dataset_loader)
    # example_input, example_target = next(iter(test_ds.take(1)))
    root_dir = os.path.dirname(os.path.realpath(__file__))
    for epoch in range(starting_epoch, epochs):
        print('{}: Starting training for epoch {}'.format(datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"),
                                                          epoch + 1))
        eval_interval_dct = _load_eval_intervals(root_dir)
        evaluation_interval = eval_interval_dct.get('evaluation_interval', evaluation_interval)
        for step, sample in enumerate(tqdm(dataset_loader, desc='Epoch {}/{}'.format(epoch + 1, epochs))):

            train_step(model, sample, optimizer, step=N * epoch + step + 1, clip_gradients=clip_gradients,
                       summary_writer=summary_writer)


            if (N * epoch + step + 1) % evaluation_interval == 0 or (N * epoch + step + 1) == N * epochs:
                print('{}: Saving the model '.format(datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")))
                checkpoint.save(file_prefix=checkpoint_prefix)
                if evaluation_fn:
                    evaluation_fn(**evaluation_fn_params, eval_step=N * epoch + step + 1)



def iterate_dataset(dataset_loader, starting_epoch, epochs):
    for epoch in range(starting_epoch, epochs):
        for step, sample in enumerate(tqdm(dataset_loader, desc='Epoch {}/{}'.format(epoch + 1, epochs))):
            print(sample[-3][-1])


# file_prefix=checkpoint_prefix,

def fit(model, train_ds, optimizer=None, clip_gradients=False, num_of_steps=None, summary_writer=None, verbose=False,
        evaluation_fn=None, evaluation_interval=-1, evaluation_fn_params=None):
    if evaluation_fn_params is None:
        evaluation_fn_params = {}
    if num_of_steps is None:
        num_of_steps = len(train_ds)
    # for step, sample in train_ds.repeat().take(num_of_steps).enumerate():

    for step, sample in enumerate(tqdm(train_ds.repeat().take(num_of_steps), total=num_of_steps, desc='Training',
                                       disable=verbose)):
        if verbose:
            start = time.time()
        train_step(model=model, input_sample=sample, optimizer=optimizer,
                   clip_gradients=clip_gradients, step=step + 1, summary_writer=summary_writer)
        # if (step + 1) % 2 == 0 and verbose:
        #     print('Time taken for step {} : {} sec\n'.format(step + 1, time.time() - start))
        # if evaluation_fn and (step + 1) % evaluation_interval == 0:
        #     evaluation_fn(**evaluation_fn_params, step=step)
        # if (step + 1) % 4 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)


def fit_v2(model, train_ds, optimizer=None, clip_gradients=False, num_of_steps=None, summary_writer=None,
           checkpoint=None, checkpoint_prefix=None, verbose=False,
           evaluation_fn=None, evaluation_interval=-1, evaluation_fn_params=None):
    if evaluation_fn_params is None:
        evaluation_fn_params = {}
    if num_of_steps is None:
        num_of_steps = len(train_ds)
    # for step, sample in train_ds.repeat().take(num_of_steps).enumerate():
    epochs = num_of_steps // evaluation_interval + (1 if num_of_steps % evaluation_interval > 0 else 0)
    print('epochs: {}'.format(epochs))
    print('evaluation interval: {}'.format(evaluation_interval))
    for epoch in range(epochs):

        remaining_steps = num_of_steps - epoch * evaluation_interval
        evaluation_interval = remaining_steps if remaining_steps < evaluation_interval else evaluation_interval
        for step, sample in enumerate(
                tqdm(train_ds.repeat().take(evaluation_interval), total=evaluation_interval, desc='Training',
                     disable=verbose)):
            # if verbose:
            #     start = time.time()
            train_step(model=model, input_sample=sample, optimizer=optimizer,
                       clip_gradients=clip_gradients, step=step + 1, summary_writer=summary_writer)
            # if (step + 1) % 2 == 0 and verbose:
            #     print('Time taken for step {} : {} sec\n'.format(step + 1, time.time() - start))
        if evaluation_fn:
            evaluation_fn(**evaluation_fn_params, step=evaluation_interval * epoch + step)
        # if (step + 1) % 4 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)


def evaluate_from_torch_dataloader(valid_loader, model, valid_ds, eval_step, experiment_name,
                                   save_rpn_predictions=False, calculate_only_proposals=False):
    model.eval()
    for step, sample in enumerate(tqdm(valid_loader, desc='Evaluating for step {}'.format(eval_step),
                                       total=len(valid_ds), disable=False)):
        # start = time.time()
        *outputs, loss = model(*sample[:-1], enable_path_drop=False, training=False)
        predictions_to_kitti(outputs, sample[-1], sample[4], sample[-2], experiment_name=experiment_name,
                             valid_ds=valid_ds, current_step=eval_step, save_rpn_predictions=save_rpn_predictions,
                             calculate_only_proposals=calculate_only_proposals)
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # call_evaluation_script_05(root_dir=root_dir, results_parent_folder='results_tf', experiment_name=experiment_name,
    #                     annotation_dir=valid_ds.annotation_dir, step=eval_step)
    calculate_map_async(root_dir=root_dir, results_parent_folder='results_tf', experiment_name=experiment_name,
                        annotation_dir=valid_ds.annotation_dir, step=eval_step)
    # print('Time taken for step {} : {} sec\n'.format(step, time.time() - start))


def evaluate(valid_loader, model, valid_ds, experiment_name, verbose=False, step=0, save_rpn_predictions=False):
    valid_ds_len = len(valid_ds)
    # start = time.time()
    for sample_idx, sample in enumerate(tqdm(valid_loader.take(valid_ds_len), total=valid_ds_len,
                                             desc='Evaluating for step {}'.format(step),
                                             disable=verbose)):
        # if verbose:
        #     start = time.time()
        *outputs, loss = model(*sample[:-1], enable_path_drop=False, training=False)
        predictions_to_kitti(outputs, sample[-1], sample[4], sample[-2], experiment_name=experiment_name,
                             valid_ds=valid_ds, save_rpn_predictions=save_rpn_predictions)
        # if verbose:
        #     print('Time taken for step {} : {} sec\n'.format(sample_idx, time.time() - start))
    root_dir = os.path.dirname(os.path.realpath(__file__))
    calculate_map_async(root_dir=root_dir, results_parent_folder='results_tf', experiment_name=experiment_name,
                        annotation_dir=valid_ds.annotation_dir, step=step)
    # calculate_map(root_dir=root_dir, results_parent_folder='results_tf', experiment_name=experiment_name,
    #               annotation_dir=valid_ds.annotation_dir, step=step)

    # call_evaluation_script(root_dir, results_parent_folder='results_tf', experiment_name=experiment_name,
    #                        annotation_dir=valid_ds.annotation_dir, step=step)


def evaluate_v2(root_dir, results_parent_folder, experiment_name, annotation_dir, checkpoint_to_evaluate=-1, step=0,
                evaluation_script_to_use='', recall=False):
    current_results_folder = os.path.join(root_dir, results_parent_folder, experiment_name, 'val', 'kitti')
    available_predictions_for_evaluation = glob.glob(current_results_folder + '/data_*')
    print('available checkpoints for evaluation: ', available_predictions_for_evaluation)
    # checkpoint_to_evaluate = available_predictions_for_evaluation[checkpoint_to_evaluate]
    # step = checkpoint_to_evaluate.split('_')[-1]
    print('step:', step)
    if evaluation_script_to_use == '05':
        if recall:
            call_evaluation_script_05_recall(root_dir, results_parent_folder=results_parent_folder,
                                             experiment_name=experiment_name, step=step, annotation_dir=annotation_dir)
        else:
            result_df = call_evaluation_script_05(root_dir, results_parent_folder=results_parent_folder,
                                                  experiment_name=experiment_name, step=step,
                                                  annotation_dir=annotation_dir,
                                                  return_df=True)
    else:
        if recall:
            call_evaluation_script_recall(root_dir, results_parent_folder=results_parent_folder,
                                          experiment_name=experiment_name, step=step, annotation_dir=annotation_dir,
                                          return_df=True)
        else:
            call_evaluation_script(root_dir, results_parent_folder=results_parent_folder,
                                   experiment_name=experiment_name, step=step,
                                   annotation_dir=annotation_dir,
                                   return_df=True)

    # print()
    # print(result_df)


def my_gen(dataset):
    for sample in dataset:
        yield sample


# @tf.function
def create_dataloader(input_dataset, shuffle=False, collate_function=None, prefetch=False,
                      num_parallel_calls=tf.data.AUTOTUNE, num_parallel_calls_prefetch=tf.data.AUTOTUNE):
    # returns a tf dataset that is like a torch dataloader
    dataloader = tf.data.Dataset.from_generator(lambda: my_gen(input_dataset), output_signature=(
        tf.TensorSpec(shape=(None, None, 3)),  # image feature maps
        tf.TensorSpec(shape=(None, None, 6)),  # bev feature maps
        tf.TensorSpec(shape=(None, 4)),  # anchors_gt_bev
        tf.TensorSpec(shape=(None, 6)),  # filtered_anchors
        tf.TensorSpec(shape=(None, 4)),  # bev_anchors
        tf.TensorSpec(shape=(None, 4)),  # img_anchors
        tf.TensorSpec(shape=(None,)),  # ious
        tf.TensorSpec(shape=(None, 6)),  # offsets
        tf.TensorSpec(shape=(None, 2)),  # objectness
        tf.TensorSpec(shape=(None,)),  # mask
        tf.TensorSpec(shape=(None, 4)),  # rect_matrix
        tf.TensorSpec(shape=(None, 1)),  # label_cls_ids
        tf.TensorSpec(shape=(None, 7)),  # labels
        tf.TensorSpec(shape=(4,)),  # ground_plane
        tf.TensorSpec(shape=(3,)),  # image_shape
        tf.TensorSpec(shape=(), dtype=tf.string)  # sample_name
    ))
    if collate_function:
        dataloader = dataloader.map(collate_function, num_parallel_calls=num_parallel_calls)
    if prefetch:
        dataloader = dataloader.prefetch(num_parallel_calls_prefetch)

    if shuffle:
        dataloader = dataloader.shuffle(len(input_dataset))

    return dataloader


def create_dataloader_v2(input_dataset, shuffle=False, collate_function=None, prefetch=False,
                         num_parallel_calls=tf.data.AUTOTUNE, num_parallel_calls_prefetch=tf.data.AUTOTUNE):
    # returns a tf dataset that is like a torch dataloader
    dataloader = tf.data.Dataset.from_generator(lambda: my_gen(input_dataset), output_signature=(
        tf.TensorSpec(shape=(1, None, None, 3)),  # image feature maps
        tf.TensorSpec(shape=(1, None, None, 6)),  # bev feature maps
        (tf.TensorSpec(shape=(None, 5)),  # img_anchors
         tf.TensorSpec(shape=(None, 5)),  # bev_anchors
         tf.TensorSpec(shape=(None, 6))),  # filtered_anchors
        tf.TensorSpec(shape=(None,)),  # mask
        tf.TensorSpec(shape=(None, 4)),  # rect_matrix
        tf.TensorSpec(shape=(4,)),  # ground_plane
        # tf.TensorSpec(shape=(None,)),  # ious
        (tf.TensorSpec(shape=(None, 6)),  # offsets
         tf.TensorSpec(shape=(None, 2)),  # objectness
         tf.TensorSpec(shape=(None, 4)),  # anchors_gt_bev
         tf.TensorSpec(shape=(None, 7)),  # labels
         tf.TensorSpec(shape=(None, 1))),  # label_cls_ids
        tf.TensorSpec(shape=(1, 1, 3)),  # image_shape
        tf.TensorSpec(shape=(), dtype=tf.string)  # sample_name
    ))
    if collate_function:
        dataloader = dataloader.map(collate_function, num_parallel_calls=num_parallel_calls)
    if prefetch:
        dataloader = dataloader.prefetch(num_parallel_calls_prefetch)

    if shuffle:
        dataloader = dataloader.shuffle(len(input_dataset))

    return dataloader


def predictions_to_kitti(predictions, sample_name, calib_dict, image_shape, experiment_name, current_step=0,
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

    final_prediction_array[:, [-2, -1]] = np.stack([max_scores, pred_class], axis=1)


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


def test_dataset_s(train_ds, num_of_steps, verbose=False):
    for step, sample in enumerate(tqdm(train_ds.repeat().take(num_of_steps), total=num_of_steps,
                                       desc='Training', disable=verbose)):
        # r = 2 * sample[0]
        pass


def test_torch_dataset_s(dataset_loader, epochs):
    N = len(dataset_loader)
    # example_input, example_target = next(iter(test_ds.take(1)))
    for epoch in range(epochs):
        for step, sample in enumerate(tqdm(dataset_loader)):
            # r = 2 * sample[0]
            pass


def benchmark_dataset(train_ds, num_iter=None):
    import tensorflow_datasets as tfds
    tfds.benchmark(train_ds, num_iter=num_iter)


def get_torch_train_dataloader(dataset_config, dataset_class=KITTIDatasetTf, collate_fn=None):
    # print('SHUFFLE VALUE IS SET TO FALSE! CHANGE IT WHEN TRAINING')
    if collate_fn is None:
        collate_fn = getattr(dataset_class, 'collate_fn')
        if 'Tf' in dataset_class.__name__:
            collate_fn = getattr(dataset_class, 'collate_fn_tf')
    train_ds_torch = dataset_class(config=dataset_config)
    train_loader_torch = DataLoader(train_ds_torch, batch_size=1, shuffle=True, num_workers=0,  # pin_memory=True,
                                    collate_fn=collate_fn)
    return train_loader_torch, train_ds_torch


def get_torch_val_dataloader(dataset_config, dataset_class=KITTIDatasetTf, collate_fn=None):
    if collate_fn is None:
        collate_fn = getattr(dataset_class, 'collate_fn')
        if 'Tf' in dataset_class.__name__:
            collate_fn = getattr(dataset_class, 'collate_fn_tf')
    val_ds_torch = dataset_class(config=dataset_config, mode='val')
    val_loader_torch = DataLoader(val_ds_torch, batch_size=1, shuffle=False, num_workers=0,
                                  collate_fn=collate_fn)
    return val_loader_torch, val_ds_torch


def get_tf_train_dataloader(dataset_config):
    train_ds = CustomDatasetTf(config=dataset_config)
    train_loader = create_dataloader(train_ds, shuffle=True, collate_function=CustomDatasetTf.collate_tf,
                                     num_parallel_calls=tf.data.AUTOTUNE, prefetch=True,
                                     num_parallel_calls_prefetch=tf.data.AUTOTUNE)
    return train_loader, train_ds


def get_tf_val_dataloader(dataset_config):
    valid_ds = CustomDatasetTf(config=dataset_config, mode='val')
    valid_loader = create_dataloader(valid_ds, collate_function=CustomDatasetTf.collate_tf,
                                     num_parallel_calls=tf.data.AUTOTUNE, prefetch=True)
    return valid_loader, valid_ds


def load_checkpoint(checkpoint, manager, load='', checkpoint_num=0):
    import glob
    if load == 'latest':
        print('Restoring  checkpoint with name: {}'.format(manager.latest_checkpoint))
        restore_checkpoint(checkpoint, manager.latest_checkpoint)
    elif load == 'specific':
        checkpoints = sorted([i[:-6] for i in glob.glob(manager._checkpoint_prefix + '*.index')])
        checkpoint_dict = {i.split('-')[-1]: i for i in checkpoints}
        print('Available checkpoints : ', checkpoint_dict)
        checkpoint_to_restore = checkpoint_dict.get(str(checkpoint_num))
        if checkpoint_to_restore:
            print('Restoring  checkpoint with name: {}'.format(checkpoint_to_restore))
            restore_checkpoint(checkpoint, checkpoint_to_restore)
        else:
            print('the checkpoint number that was provided does not exist')
    else:
        checkpoints = sorted([i[:-6] for i in glob.glob(manager._checkpoint_prefix + '*.index')])
        if len(checkpoints) < checkpoint_num - 1:
            checkpoint_num = len(checkpoints) - 1
        print('Available checkpoints : ', checkpoints)
        if not checkpoints:
            return
        print('Restoring  checkpoint with name: {}'.format(checkpoints[checkpoint_num]))
        restore_checkpoint(checkpoint, checkpoints[checkpoint_num])


def restore_checkpoint(checkpoint, path):
    checkpoint.restore(path)


def _load_eval_intervals(root_dir):
    file = os.path.join(root_dir, 'checkpoints_to_evaluate_tf.json')
    eval_intervals_dict = {}
    try:
        with open(file, 'r') as f:
            eval_intervals_dict = json.load(f)
    except FileNotFoundError:
        pass
    return eval_intervals_dict


def predict(model, sample):
    return model(*sample[:-1], training=False, enable_path_drop=False)


def predict_dataset(model, dataloader):
    for idx, sample in enumerate(dataloader):
        predictions = predict(model, sample)


def evaluate_v3(valid_loader, model, valid_ds, experiment_name, verbose=False, step=0):
    valid_ds_len = len(valid_ds)
    best_metrics = {'f1': 0.}
    worst_metrics = {'f1': 1}
    # start = time.time()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    result_folder = root_dir + '/results_tf/' + experiment_name + '/val'
    kitti_predictions_path = result_folder + '/kitti'
    for sample_idx, sample in enumerate(tqdm(valid_loader, desc='Evaluating for step {}'.format(step),
                                             disable=verbose)):

        bbs, predicted_labels = load_predictions_and_labels(valid_ds, sample_idx,
                                                            sample[-2][0][0],
                                                            [360, 1200],  # new shape
                                                            sample[-1][0],  # sample name
                                                            kitti_predictions_path,
                                                            step)

        metrics = calculate_metrics_per_image(sample[-1][0], bbs, predicted_labels)
        if metrics['f1'] >= best_metrics['f1']:
            best_metrics = metrics
            best_sample = sample
        if metrics['f1'] < worst_metrics['f1']:
            worst_metrics = metrics
            worst_sample = sample

    print()


def fill_metrics_df_from_dict(metrics_dict, columns, index):
    metrics = {}
    for k in columns:
        v = metrics_dict[k]
        metrics[k] = v.numpy().item()
    if not isinstance(index, list):
        index = [index]
    df = pd.DataFrame(metrics, columns=columns, index=index)
    return df


def per_image_metrics(valid_loader, valid_ds, experiment_name, step=0, verbose=False, limit=None):
    from utils.general_utils_tf import plot_predictions_and_metrics
    from utils.general_utils_tf import load_predictions_and_labels, calculate_metrics_per_image
    total = limit if limit else len(valid_ds)

    best_metrics = {'f1': 0.}
    worst_metrics = {'f1': 1}
    # start = time.time()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    result_folder = root_dir + '/results_tf/' + experiment_name + '/val'
    kitti_predictions_path = result_folder + '/kitti'
    columns = ['precision', 'recall', 'f1', 'fp', 'tp', 'fn']
    metrics_df = pd.DataFrame(columns=columns)
    for sample_idx, sample in enumerate(tqdm(valid_loader, desc='Computing metrics per image for step {}'.format(step),
                                             disable=verbose, total=total)):

        bbs, predicted_labels = load_predictions_and_labels(valid_ds, sample_idx,
                                                            sample[-2][0][0],
                                                            [360, 1200],  # new shape
                                                            sample[-1][0],  # sample name
                                                            kitti_predictions_path,
                                                            step)
        if predicted_labels is None:
            continue
        metrics = calculate_metrics_per_image(sample[-1][0], bbs, predicted_labels)
        metrics_df = metrics_df.append(fill_metrics_df_from_dict(metrics, columns, sample_idx))
        if metrics['f1'] >= best_metrics['f1']:
            best_metrics = metrics
            best_sample = sample
            # best_act = activation
        if metrics['f1'] < worst_metrics['f1']:
            worst_metrics = metrics
            worst_sample = sample
        if limit == sample_idx:
            break
    metrics_df_sorted = metrics_df.sort_values(by='f1', ascending=False)
    plot_predictions_and_metrics(best_metrics, sample_name=best_sample[-1][0], image=best_sample[0][0],
                                 epoch=step, bbs=bbs)
    plot_predictions_and_metrics(worst_metrics, sample_name=worst_sample[-1][0], image=worst_sample[0][0],
                                 epoch=step, bbs=bbs)


    print()


def get_image_metrics(valid_loader, valid_ds, experiment_name, step=0, verbose=False, limit=None):
    from utils.general_utils_tf import plot_predictions_and_metrics
    from utils.general_utils_tf import load_predictions_and_labels, calculate_metrics_per_image

    best_metrics = {'f1': 0.}
    worst_metrics = {'f1': 1}
    # start = time.time()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    result_folder = root_dir + '/results_tf/' + experiment_name + '/val'
    kitti_predictions_path = result_folder + '/kitti'
    columns = ['precision', 'recall', 'f1', 'fp', 'tp', 'fn']
    metrics_df = pd.DataFrame(columns=columns)
    for sample_idx, sample in enumerate(tqdm(valid_loader, desc='Computing metrics per image for step {}'.format(step),
                                             disable=verbose)):
        # if sample_idx == 12:
        #     print()
        bbs, predicted_labels = load_predictions_and_labels(valid_ds, sample_idx,
                                                            sample[-2][0][0],
                                                            [360, 1200],  # new shape
                                                            sample[-1][0],  # sample name
                                                            kitti_predictions_path,
                                                            step)
        if predicted_labels is None:
            continue
        metrics = calculate_metrics_per_image(sample[-1][0], bbs, predicted_labels)
        metrics_df = metrics_df.append(fill_metrics_df_from_dict(metrics, columns, sample_idx))
        if metrics['f1'] >= best_metrics['f1']:
            best_metrics = metrics
            best_sample = sample
            # best_act = activation
        if metrics['f1'] < worst_metrics['f1']:
            worst_metrics = metrics
            worst_sample = sample
        if limit == sample_idx:
            break
    return metrics_df, best_metrics, worst_metrics, best_sample, worst_sample, bbs


def get_image_metrics_torch(valid_loader, valid_ds, experiment_name, step=0, verbose=False, limit=None):
    from utils.general_utils import load_predictions_and_labels, calculate_metrics_per_image
    total = limit if limit else len(valid_ds)
    best_metrics = {'f1': 0.}
    worst_metrics = {'f1': 1}
    # start = time.time()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    result_folder = root_dir + '/results_tf/' + experiment_name + '/val'
    kitti_predictions_path = result_folder + '/kitti'
    columns = ['precision', 'recall', 'f1', 'fp', 'tp', 'fn']
    metrics_df = pd.DataFrame(columns=columns)
    for sample_idx, sample in enumerate(tqdm(valid_loader, desc='Computing metrics per image for step {}'.format(step),
                                             disable=verbose, total=total)):

        bbs, predicted_labels = load_predictions_and_labels(valid_ds, sample_idx,
                                                            sample[-2][0][0],
                                                            [360, 1200],  # new shape
                                                            sample[-1][0],  # sample name
                                                            kitti_predictions_path,
                                                            step)
        if predicted_labels is None:
            continue
        metrics = calculate_metrics_per_image(sample[-1][0], bbs, predicted_labels)
        metrics_df = metrics_df.append(fill_metrics_df_from_dict(metrics, columns, sample_idx))
        if metrics['f1'] >= best_metrics['f1']:
            best_metrics = metrics
            best_sample = sample
            # best_act = activation
        if metrics['f1'] < worst_metrics['f1']:
            worst_metrics = metrics
            worst_sample = sample
        if limit == sample_idx:
            break
    return metrics_df, best_metrics, worst_metrics, best_sample, worst_sample, bbs

# https://github.com/kujason/avod
def calculate_3d_iou(box, boxes):
    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    box_diag = np.sqrt(np.square(box[1]) +
                       np.square(box[2]) +
                       np.square(box[3])) / 2

    boxes_diag = np.sqrt(np.square(boxes[:, 1]) +
                         np.square(boxes[:, 2]) +
                         np.square(boxes[:, 3])) / 2

    dist = np.sqrt(np.square(boxes[:, 4] - box[4]) +
                   np.square(boxes[:, 5] - box[5]) +
                   np.square(boxes[:, 6] - box[6]))

    non_empty = box_diag + boxes_diag >= dist

    iou = np.zeros(len(boxes), np.float64)

    if non_empty.any():
        height_int, _ = height_metrics(box, boxes[non_empty])
        rect_int = get_rectangular_metrics(box, boxes[non_empty])

        intersection = np.multiply(height_int, rect_int)

        vol_box = np.prod(box[1:4])

        vol_boxes = np.prod(boxes[non_empty, 1:4], axis=1)

        union = vol_box + vol_boxes - intersection

        iou[non_empty] = intersection / union

    if iou.shape[0] == 1:
        iou = iou[0]

    return iou


def height_metrics(box, boxes):
    """Compute 3D height intersection and union between a box and a list of
    boxes

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return height_intersection: a numpy array containing the intersection along
    the gravity axis between the two bbs

    :return height_union: a numpy array containing the union along the gravity
    axis between the two bbs
    """
    boxes_heights = boxes[:, 2]
    boxes_centroid_heights = boxes[:, 5]

    min_y_boxes = boxes_centroid_heights - boxes_heights

    max_y_box = box[5]
    min_y_box = box[5] - box[2]

    max_of_mins = np.maximum(min_y_box, min_y_boxes)
    min_of_maxs = np.minimum(max_y_box, boxes_centroid_heights)

    offsets = min_of_maxs - max_of_mins
    height_intersection = np.maximum(0, offsets)

    height_union = np.maximum(min_y_box, boxes_centroid_heights) \
                   - np.minimum(min_y_box, min_y_boxes) - \
                   np.maximum(0, -offsets)

    return height_intersection, height_union


def get_rotated_3d_bb(boxes):
    """Compute rotated 3D bounding box coordinates.

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],...
                                         [ry, l, h, w, tx, ty, tz]]

    :return x: x coordinates of the four corners required to describe a 3D
    bounding box arranged as [[x1, x2, x3, x4],
                     [x1, x2, x3, x4],
                     ... ]

    :return z: z coordinates of the four corners required to describe a 3D
    bounding box arranged as [[z1, z2, z3, z4],
                     [z1, z2, z3, z4],
                     ... ].
    """

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    x = np.array([[]])
    z = np.array([[]])

    for i in boxes:
        rot_mat = np.array([[np.cos(i[0]), np.sin(i[0])],
                            [-np.sin(i[0]), np.cos(i[0])]])

        x_corners = np.multiply(i[1] / 2, np.array([1, 1, -1, -1]))
        z_corners = np.multiply(i[3] / 2, np.array([1, -1, -1, 1]))

        temp_coor = np.dot(rot_mat, np.array([x_corners, z_corners]))

        # At the very first iteration, initialize x
        if x.shape[1] < 1:
            x = temp_coor[:1] + i[4]
            z = temp_coor[1:2] + i[6]
        # After that, append to the existing x
        else:
            x = np.append(x, temp_coor[:1] + i[4], axis=0)
            z = np.append(z, temp_coor[1:2] + i[6], axis=0)

    if x.shape[0] == 1:
        x = x[0]
        z = z[0]

    return x, z


def get_rectangular_metrics(box, boxes):
    """ Computes the intersection of the bases of oriented 3D bounding "box"
    and a set boxes of oriented 3D bounding boxes "boxes".

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return intersection: a numpy array containing intersection between the
    base of box and all other boxes.
    """
    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    mask_res = 0.01

    x_box, z_box = get_rotated_3d_bb(box)
    max_x_box = np.max(x_box)
    min_x_box = np.min(x_box)
    max_z_box = np.max(z_box)
    min_z_box = np.min(z_box)

    x_boxes, z_boxes = get_rotated_3d_bb(boxes)

    intersection = np.zeros(np.size(boxes, 0))

    if np.size(np.shape(x_boxes)) == 1:
        x_boxes = np.array([x_boxes])
        z_boxes = np.array([z_boxes])

    for i in range(np.size(boxes, 0)):
        x_i = x_boxes[i, :]
        z_i = z_boxes[i, :]
        test = max_x_box < np.min(x_i) or np.max(x_i) < min_x_box \
               or max_z_box < np.min(z_i) or np.max(z_i) < min_z_box

        if test:
            continue

        x_all = np.append(x_box, x_i)
        z_all = np.append(z_box, z_i)
        maxs = np.array([np.max(x_all), np.max(z_all)])
        mins = np.array([np.min(x_all), np.min(z_all)])

        mask_dims = np.int32(np.ceil((maxs - mins) / mask_res))

        mask_box_x = (x_box - mins[0]) / mask_res
        mask_box_z = (z_box - mins[1]) / mask_res
        mask_i_x = (x_i - mins[0]) / mask_res
        mask_i_z = (z_i - mins[1]) / mask_res
        # Drawing a binary image of the base of the two bounding boxes.
        # Then compute the element wise and of the two images to get the intersection.
        # Minor precision loss due to discretization.
        img = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img, 'L')
        rect_coordinates = np.reshape(np.transpose(np.array([mask_box_x,
                                                             mask_box_z])), 8)
        rect_coordinates = np.append(rect_coordinates, rect_coordinates[0:2])
        draw.polygon(rect_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_box = np.asarray(img)

        img2 = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img2, 'L')
        i_coordinates = np.reshape(np.transpose(np.array([mask_i_x,
                                                          mask_i_z])), 8)
        i_coordinates = np.append(i_coordinates, i_coordinates[0:2])
        draw.polygon(i_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_i = np.asarray(img2)

        mask_intersection = np.logical_and(mask_box, mask_i)
        intersection[i] = min(100, np.size(np.flatnonzero(
            mask_intersection)) * np.square(mask_res))

    if intersection.shape[0] == 1:
        intersection = intersection[0]

    return intersection


def box_3d_to_3d_iou_format(boxes_3d):
    """ Returns a numpy array of 3d box format for iou calculation
    Args:
        boxes_3d: list of 3d boxes
    Returns:
        new_anchor_list: numpy array of 3d box format for iou
    """
    boxes_3d = np.asarray(boxes_3d)
    # fc.check_box_3d_format(boxes_3d)

    iou_3d_boxes = np.zeros([len(boxes_3d), 7])
    iou_3d_boxes[:, 4:7] = boxes_3d[:, 0:3]
    iou_3d_boxes[:, 1] = boxes_3d[:, 3]
    iou_3d_boxes[:, 2] = boxes_3d[:, 4]
    iou_3d_boxes[:, 3] = boxes_3d[:, 5]
    iou_3d_boxes[:, 0] = boxes_3d[:, 6]

    return iou_3d_boxes


def calculate_sample_tp(gt_boxes, proposals, iou_thresh=0.5):
    gt_boxes_iou = box_3d_to_3d_iou_format(gt_boxes)
    proposals_iou = box_3d_to_3d_iou_format(proposals)
    # box_num = len(gt_boxes)
    tp = 0
    ious = np.zeros((len(gt_boxes_iou), len(proposals_iou)))
    # fn = 0
    for idx, gt_box_iou in enumerate(gt_boxes_iou):
        ious[idx] = calculate_3d_iou(gt_box_iou, proposals_iou)
        # if iou >= iou_thresh:
        #     tp += 1
    # max_ious = np.amax(ious, axis=0)
    max_ids = np.argmax(ious, axis=0)
    for i in range(len(gt_boxes_iou)):
        current_max_ids = max_ids == i
        if ious[i, :][current_max_ids].any():
            if ious[i, :][current_max_ids].max() >= iou_thresh:
                tp += 1
    return tp


def split_annotations_to_diff_categs(annotations):
    return None, None, None


def calculate_tp_using_ann(ann, proposals):
    # ids = set(ann['idx'])
    # obj_num = len(ann)
    # for i in ids:
    current_labels = ann.loc[:, ['x', 'y', 'z', 'l', 'w', 'h', 'ry']].values.astype(np.float32)

    tp = calculate_sample_tp(current_labels, proposals)
    return tp


def calculate_proposal_recall(proposals_dir, easy_ann, mod_ann, hard_ann):
    proposal_files = glob.glob(proposals_dir + '/*.txt')
    tp_easy = 0
    tp_mod = 0
    tp_hard = 0
    num_easy = len(easy_ann)
    num_mod = len(mod_ann)
    num_hard = len(hard_ann)
    result_columns = ['sample_name', 'tp', 'num_samples', 'recall']
    df_results_easy = pd.DataFrame(columns=result_columns)
    df_results_mod = pd.DataFrame(columns=result_columns)
    df_results_hard = pd.DataFrame(columns=result_columns)
    for i, proposal_file in tqdm(enumerate(proposal_files), total=len(proposal_files), desc="evaluating proposals"):
        sample_name = proposal_file.split('/')[-1].split('.')[0]
        # print(sample_name)
        proposals = pd.read_csv(proposal_file).values.astype(np.float32)
        c_e_ann = easy_ann[easy_ann['image_name'] == sample_name + '.png']
        c_m_ann = mod_ann[mod_ann['image_name'] == sample_name + '.png']
        c_h_ann = hard_ann[hard_ann['image_name'] == sample_name + '.png']
        if len(c_e_ann) > 0:
            c_easy_tp = calculate_tp_using_ann(c_e_ann, proposals)
            tp_easy += c_easy_tp
            c_df_easy = pd.DataFrame(columns=result_columns)
            c_df_easy.loc[sample_name, result_columns] = [sample_name, c_easy_tp, len(c_e_ann),
                                                          c_easy_tp / len(c_e_ann)]
            # print(c_df_easy)
            df_results_easy = df_results_easy.append(c_df_easy)
        if len(c_m_ann) > 0:
            c_tp_mod = calculate_tp_using_ann(c_m_ann, proposals)
            tp_mod += c_tp_mod
            c_df_mod = pd.DataFrame(columns=result_columns)
            c_df_mod.loc[sample_name, result_columns] = [sample_name, c_tp_mod, len(c_m_ann), c_tp_mod / len(c_m_ann)]
            df_results_mod = df_results_mod.append(c_df_mod)
        if len(c_h_ann) > 0:
            c_tp_hard = calculate_tp_using_ann(c_h_ann, proposals)
            tp_hard += c_tp_hard
            c_df_hard = pd.DataFrame(columns=result_columns)
            c_df_hard.loc[sample_name, result_columns] = [sample_name, c_tp_hard, len(c_h_ann),
                                                          c_tp_hard / len(c_h_ann)]
            df_results_hard = df_results_hard.append(c_df_hard)
    easy_recall = tp_easy / num_easy
    mod_recall = tp_mod / num_mod
    hard_recall = tp_hard / num_hard
    print(df_results_easy)
    print(df_results_mod)
    print(df_results_hard)
    print("Recall:")
    print('Easy: {}'.format(easy_recall))
    print('Moderate: {}'.format(mod_recall))
    print('Hard: {}'.format(hard_recall))
    print()
