import os
import math
import torch
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiprocessing import Pool
from matplotlib.collections import PatchCollection
from utils.box_utils_tf import calculate_iou, convert_orientation_to_angle, convert_box_to_4cp_format, \
    convert_4cp_to_offsets, call_proper_method

functs = []


def load_annotations_par(files):
    cores = multiprocessing.cpu_count()
    p = Pool(cores)
    size = len(files)
    chunk_size = math.ceil(size / cores)
    chunks = [files[i:i + chunk_size] for i in range(0, size, chunk_size)]
    res = pd.concat(p.map(load_annotations2, chunks))
    p.close()
    p.join()
    return res


def load_annotations2(files):
    columns = ['idx', 'class', 'truncation', 'occlusion', 'alpha', 'minx', 'miny', 'maxx', 'maxy', 'h', 'w', 'l',
               'x',
               'y', 'z', 'ry', 'image_name']

    df = pd.DataFrame(columns=columns)
    # cols_to_keep = [0, 1,2,34,4, 5, 6, 7]
    for i, file in enumerate(files):
        temp_df = pd.read_csv(file, sep=" ", header=None)
        temp_df['image_name'] = file.split('/')[-1].split('.')[0] + '.png'
        temp_df.columns = columns[1:]
        temp_df['idx'] = i
        df = df.append(temp_df)
    return df


def create_point_filter(extents, point_cloud, ground_plane, offset):
    x_extents, y_extents, z_extents = extents
    filter = (point_cloud[0] > x_extents[0]) & \
             (point_cloud[0] < x_extents[1]) & \
             (point_cloud[1] > y_extents[0]) & \
             (point_cloud[1] < y_extents[1]) & \
             (point_cloud[2] > z_extents[0]) & \
             (point_cloud[2] < z_extents[1])

    off_plane = ground_plane + [0, 0, 0, -offset]
    point_cloud_pad = np.ones((point_cloud.shape[0] + 1, point_cloud.shape[1]))
    point_cloud_pad[:3, :] = point_cloud
    plane_filter = np.dot(off_plane, point_cloud_pad) < 0
    point_filter = np.logical_and(filter, plane_filter)

    return point_filter


def filter_point_cloud(extents, point_cloud, ground_plane):
    height_l, height_h = 0.2, 2.0

    offset_filter = create_point_filter(extents, point_cloud=point_cloud, ground_plane=ground_plane, offset=height_h)
    road_filter = create_point_filter(extents, point_cloud=point_cloud, ground_plane=ground_plane, offset=height_l)
    return np.logical_xor(offset_filter, road_filter)


def create_mini_batch_mask_rpn(gt_anchors_bev, anchors_bev, pos_range=None, neg_range=None, size=512, ratio=0.5):
    if pos_range is None:
        pos_range = [0.5, 1.0]
    if neg_range is None:
        neg_range = [0.0, 0.3]

    ious = calculate_iou(gt_anchors_bev, anchors_bev)
    max_ious, max_ids = torch.max(ious, dim=0)
    mask, pos_mask = get_iou_mask(max_ious, neg_range, pos_range, num_of_elements=size, pos_neg_ratio=ratio)
    return mask, pos_mask


def create_mini_batch_mask(gt_anchors_bev, anchors_bev, cls_labels, size=1024, ratio=0.5, pos_range=None, neg_range=None):

    ious = tf.reshape(calculate_iou(gt_anchors_bev, anchors_bev), (-1, 1024))

    max_ids = tf.math.argmax(ious, axis=0)
    max_ious = tf.math.reduce_max(ious, axis=0)


    mask, pos_mask = get_iou_mask(max_ious, neg_range, pos_range, num_of_elements=size, pos_neg_ratio=ratio)

    gt_ids = tf.boolean_mask(max_ids, mask)
    masked_labels = tf.gather(cls_labels, gt_ids)
    # remove ignored anchors i.e. those who have an iou score that lies between the upper and lower bound of negative
    # and positive ranges respectively
    mask_pos = tf.boolean_mask(pos_mask, mask)
    cls_mask = tf.cast(mask_pos, tf.int32) * tf.cast(tf.squeeze(masked_labels), tf.int32)

    # gt_ids shows which label has the highest iou wrt each anchor(rpn proposals)
    return mask, cls_mask, gt_ids


@call_proper_method
def create_one_hot(cls_masked, neg_val=0.001, num_classes=2): ...


def create_target_angle(label_boxes, gt_masked):
    gt_or_masked = tf.gather(label_boxes[:, 6], gt_masked)
    return convert_orientation_to_angle(gt_or_masked)


def create_target_offsets(gt_box_masked, proposals_4cp_masked, ground_plane):
    gt_anchors_4cp = convert_box_to_4cp_format(gt_box_masked, ground_plane)
    offsets_gt_masked = convert_4cp_to_offsets(proposals_4cp_masked, gt_anchors_4cp)
    return offsets_gt_masked


def create_regression_offsets(anchors: np.ndarray, gt_label: np.ndarray) -> np.ndarray:
    """
    Given the labels to axis aligned format and the filtered anchors it creates the regression targets
    """
    offsets = np.zeros((len(anchors), 6), dtype=np.float32)
    offsets[:, 0:3] = (gt_label[:, 0:3] - anchors[:, 0:3]) / anchors[:, 3:]
    offsets[:, 3:] = np.log(gt_label[:, 3:] / anchors[:, 3:])
    return offsets


def get_iou_mask(ious, neg_range, pos_range, num_of_elements=1024, pos_neg_ratio=0.5, **kwargs):
    background_negative = ious < neg_range[1]
    pos = ious >= pos_range[0]
    # noinspection PyUnresolvedReferences
    if tf.is_tensor(pos):
        mask, pos_ids = subsample_array(tf.squeeze(pos), tf.squeeze(background_negative),
                                        num_of_elements, pos_neg_ratio)
    else:
        mask, pos_ids = subsample_array(pos.squeeze(), background_negative.squeeze(), num_of_elements, pos_neg_ratio)
    return mask, pos_ids


def subsample_array(pos_mask, neg_mask, num_of_elements, fraction):
    if tf.is_tensor(pos_mask):
        # if pos_mask.device.type != 'cpu':
        return subsample_tensor(pos_mask, neg_mask, num_of_elements, fraction)

    mask = np.full((len(pos_mask)), False)

    # noinspection PyTypeChecker
    num_of_pos = min(int(fraction * num_of_elements), pos_mask.sum())
    pos_indices_shuffled = np.where(pos_mask)[0]
    np.random.shuffle(pos_indices_shuffled)
    pos_indices = pos_indices_shuffled[:num_of_pos]
    # indices_to_select_from = np.where(pos_mask)[0]
    num_of_neg = num_of_elements - num_of_pos
    neg_indices_shuffled = np.where(neg_mask)[0]
    np.random.shuffle(neg_indices_shuffled)
    neg_indices = neg_indices_shuffled[:num_of_neg]
    mask[pos_indices] = True
    mask[neg_indices] = True
    pos_mask[~pos_indices] = False
    return mask, pos_mask


def subsample_tensor(pos_mask, neg_mask, num_of_elements, fraction):
    mask = tf.fill(dims=pos_mask.shape, value=False)
    pos_indices = tf.where(pos_mask)
    pos_indices = tf.random.shuffle(pos_indices)
    not_pos_indices = tf.where(~pos_mask)
    neg_indices = tf.where(neg_mask)
    neg_indices = tf.random.shuffle(neg_indices)
    mask = tf.tensor_scatter_nd_update(mask, pos_indices, tf.fill(dims=pos_indices.shape[0], value=True))
    mask = tf.tensor_scatter_nd_update(mask, neg_indices, tf.fill(dims=neg_indices.shape[0], value=True))
    pos_mask = tf.tensor_scatter_nd_update(pos_mask, not_pos_indices, tf.fill(dims=not_pos_indices.shape[0], value=False))

    return mask, pos_mask


def _expanduser_(funct):
    def wrapper(*args):
        return funct(os.path.expanduser(*args))

    return wrapper


@_expanduser_
def exists(path=''):
    return os.path.exists(path)


def _display_(num_plots=1, figsize=(15, 15), titles=[]):
    fig, ax = plt.subplots(num_plots, figsize=figsize)
    # num_plots > 1 and
    if len(titles) < num_plots:
        titles = num_plots * [None]
    for i, (func, args, kwargs) in enumerate(functs):
        if num_plots > 1:
            ax[i].title.set_text(titles[i])
            kwargs.update(ax=ax[i])
            func(*args, **kwargs)
        else:
            ax.title.set_text(titles[i])
            kwargs.update(ax=ax)
            ax = func(*args, **kwargs)
    plt.show()


def register_display(func):
    def wrapper(*args, **kwargs):
        global functs
        if len(functs) == 2:
            print('Clearing registered functions')
            functs = []
        functs.append((func, args, kwargs))
        # return ax

    return wrapper


def _display_image_with_2d_bb_(ax, image, bbs, categs, _plot=True, scores=None, colors=None, fontsize=None):
    bbs_ = []
    i = 0
    if fontsize is None:
        fontsize = 'large'
    noc = len(set(categs))
    if colors is None:
        colors = np.random.rand(noc, 3)
    color_mapping = {colname: colors[i] for i, colname in enumerate(list(set(categs)))}
    for i in range(bbs.shape[0]):

        bl = bbs[i, [0, 1]]
        h = bbs[i, 2] - bbs[i, 0]

        w = bbs[i, 3] - bbs[i, 1]

        bbs_.append(patches.Rectangle(tuple(bl), h, w, edgecolor=color_mapping[categs[i]], facecolor='none'))
        if scores is not None:
            ax.text(bbs[i, 2], bbs[i, -1], str(round(scores[i], 4)),
                    horizontalalignment='right',
                    verticalalignment='bottom', color=color_mapping[categs[i]], size=fontsize)

    ax.imshow(image)
    edgecolor = None
    if noc == 1:
        edgecolor = color_mapping[categs[i]]
    ax.add_collection(PatchCollection(bbs_, linewidth=2, match_original=True))
    # ax.add_collection(PatchCollection(bbs_, linewidth=2, edgecolor=edgecolor, facecolor='none'))

    if _plot:
        plt.show()
    return ax


def _display_image_with_3d_bb_(ax, image, bbs, categs, _plot=True, colors=None):
    noc = len(set(categs))
    colors = np.random.rand(noc, 3)
    color_mapping = {colname: colors[i] for i, colname in enumerate(list(set(categs)))}
    h, w, _ = image.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.imshow(image)
    front_side = [0, 1, 5, 4]
    left_side = [1, 2, 6, 5]
    back_side = [2, 3, 7, 6]
    right_side = [3, 0, 4, 7]
    for bb_idx in range(bbs.shape[1]):
        bb = bbs[:, bb_idx, :]
        ax.plot(np.append(bb[0, front_side], bb[0, front_side[0]]),
                np.append(bb[1, front_side], bb[1, front_side[0]]), linewidth=1,
                color=color_mapping[categs[bb_idx]])
        ax.plot(np.append(bb[0, left_side], bb[0, left_side[0]]),
                np.append(bb[1, left_side], bb[1, left_side[0]]), linewidth=1, color=color_mapping[categs[bb_idx]])
        ax.plot(np.append(bb[0, back_side], bb[0, back_side[0]]),
                np.append(bb[1, back_side], bb[1, back_side[0]]), linewidth=1, color=color_mapping[categs[bb_idx]])
        ax.plot(np.append(bb[0, right_side], bb[0, right_side[0]]),
                np.append(bb[1, right_side], bb[1, right_side[0]]), linewidth=1,
                color=color_mapping[categs[bb_idx]])
    if _plot:
        plt.show()
    return ax


def display_image_with_bb(bb_type='2d', **kwargs):
    if bb_type == '2d':
        return _display_image_with_2d_bb_(**kwargs)
    elif bb_type == '3d':
        return _display_image_with_3d_bb_(**kwargs)
