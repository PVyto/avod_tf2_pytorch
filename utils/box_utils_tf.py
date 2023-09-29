import torch
import numpy as np
import tensorflow as tf
from typing import Union
from abc import ABC, abstractmethod
from functools import wraps
# from torchvision.ops.boxes import box_iou
from utils.general_utils import multiply, create_rotation_matrix

T = Union[tf.Tensor, np.ndarray]


class BoxUtilsAbstract(ABC):
    @staticmethod
    @abstractmethod
    def box_2d_from_corners(corners): ...

    @staticmethod
    @abstractmethod
    def offset_predictions_to_anchors(predictions, anchors): ...

    @classmethod
    @abstractmethod
    def proposals_to_4c_box_al(cls, proposals, offset_scores, ground_plane): ...

    @staticmethod
    @abstractmethod
    def convert_box_to_axis_al_format(boxes_3d, ortho_rotate=False): ...

    @staticmethod
    @abstractmethod
    def convert_anchors_to_box_format(anchors, get_longest_dim_as_len=False): ...

    @staticmethod
    @abstractmethod
    def convert_4cp_to_box_format(boxes_4cp, gp): ...

    @classmethod
    @abstractmethod
    def convert_box_to_4cp_format(cls, bxs_3d, gp): ...

    @staticmethod
    @abstractmethod
    def convert_orientation_to_angle(or_v): ...

    @staticmethod
    @abstractmethod
    def convert_angle_to_orientation(a_vec): ...

    @staticmethod
    @abstractmethod
    def extract_box_info(vec, vec_norm, points, mid_point): ...

    @staticmethod
    @abstractmethod
    def extract_corners_from_label(label_box: T) -> T: ...

    @staticmethod
    @abstractmethod
    def extract_corners_from_axis_aligned(axis_aligned_boxes): ...

    @classmethod
    @abstractmethod
    def project_box_to_img(cls, box, calib_dict, image_shape): ...

    @staticmethod
    @abstractmethod
    def project_lidar_to_img(pts, calib_dict): ...

    # TODO:CHANGE THE NAME OF THE FUNCTION BELOW; CORNERS ARE EXTRACTED FROM AXIS ALIGNED BOXES WHICH THEN ARE
    #  PROJECTED TO IMAGE SPACE
    @classmethod
    @abstractmethod
    def project_axis_aligned_to_img(cls, axis_aligned_anchors, image_shape, calib_dict=None, normalize=False): ...

    @staticmethod
    @abstractmethod
    def project_to_bev(axis_aligned_anchors: T, area_extents: T, normalize: bool = False, **kwargs) -> T: ...

    @staticmethod
    @abstractmethod
    def calculate_iou(box1, box2): ...

    @staticmethod
    @abstractmethod
    def create_one_hot(cls_masked, neg_val=0.001, num_classes=2): ...


class BoxUtilsTensor(BoxUtilsAbstract):

    @staticmethod
    def box_2d_from_corners(corners):
        # transpose to correctly extract for number of boxes>1
        xmin, ymin = torch.amin(corners, dim=0).T
        xmax, ymax = torch.amax(corners, dim=0).T
        # transpose for the case boxes>1 and reshape when boxes==1
        return torch.stack([xmin, ymin, xmax, ymax]).T.reshape(-1, 4)

    @staticmethod
    def offset_predictions_to_anchors(predictions, anchors):
        # pred_anchors = tf.zeros((len(predictions), 6))
        xyz = predictions[:, 0:3] * anchors[:, 3:] + anchors[:, 0:3]
        whl = tf.math.exp(tf.math.log(anchors[:, 3:]) + predictions[:, 3:])

        return tf.concat([xyz, whl], axis=1)

    @classmethod
    def proposals_to_4c_box_al(cls, proposals, offset_scores, ground_plane):
        proposals_box_format = cls.convert_anchors_to_box_format(proposals, True)
        proposals_4cp_format = cls.convert_box_to_4cp_format(proposals_box_format, ground_plane)
        # test convert_4cp_to_box_format

        predictions_4c = convert_offsets_to_4cp_format(proposals_4cp_format, offset_scores)
        predictions_box = cls.convert_4cp_to_box_format(predictions_4c, ground_plane)
        pred_anchors = cls.convert_box_to_axis_al_format(predictions_box, True)

        return predictions_4c, predictions_box, pred_anchors, proposals_4cp_format

    @staticmethod
    def convert_box_to_axis_al_format(boxes_3d, ortho_rotate=False):
        """
        converts box to axis aligned format:
        [x, y, z, l, w, h, ry] -> [x, y, z, dim_x, dim_y, dim_z]
        """

        x, y, z, l, w, dy, ry = tf.unstack(boxes_3d, axis=1)
        # h stays the same i.e h = dim_y
        # calculate dx, dz given l, w, ry
        if ortho_rotate:
            half_pi = np.pi / 2
            ry = tf.math.round(ry / half_pi) * half_pi

        cos_ry = tf.math.abs(tf.math.cos(ry))
        sin_ry = tf.math.abs(tf.math.sin(ry))
        # dx
        dx = l * cos_ry + w * sin_ry
        # dz
        dz = w * cos_ry + l * sin_ry

        return tf.stack([x, y, z, dx, dy, dz], axis=1)

    @staticmethod
    def convert_anchors_to_box_format(anchors, get_longest_dim_as_len=False):
        x, y, z, l, h, w = tf.unstack(anchors, axis=1)
        ry = tf.zeros(len(anchors))
        if get_longest_dim_as_len:
            indices_to_swap = tf.where(w > l)
            l_copy = l

            try:
                l = tf.tensor_scatter_nd_update(l, indices_to_swap, tf.gather(w, tf.squeeze(indices_to_swap)))
                w = tf.tensor_scatter_nd_update(w, indices_to_swap, tf.gather(l_copy, tf.squeeze(indices_to_swap)))
            except Exception as e:
                print('Exception occurred at convert_anchors_to_box_format')
                print(str(e))
                print(indices_to_swap.shape)
            ry = tf.tensor_scatter_nd_update(ry, indices_to_swap, tf.fill(indices_to_swap.shape[0], -np.pi / 2))
        return tf.stack([x, y, z, l, w, h, ry], axis=1)

    @classmethod
    def convert_4cp_to_box_format(cls, boxes_4cp, gp):
        # extracting corners
        corners = tf.reshape(boxes_4cp[:, :8], (-1, 2, 4))
        mid_pts = (corners[:, :, :3] + corners[:, :, 1:4]) / 2
        mid_pts = tf.concat([mid_pts, (corners[:, :, 0:1] + corners[:, :, 3:4]) / 2], axis=2)

        vec34_12 = mid_pts[:, :, 0] - mid_pts[:, :, 2]
        vec34_12_norm = tf.norm(vec34_12, axis=1)

        vec23_14 = mid_pts[:, :, 3] - mid_pts[:, :, 1]
        vec23_14_norm = tf.norm(vec23_14, axis=1)

        c_vec34_12, vec34_12_len, vec34_12_w, vec34_12_ry = cls.extract_box_info(vec34_12, vec34_12_norm,
                                                                                 corners,
                                                                                 mid_point=mid_pts[:, :, 2])

        c_vec23_14, vec23_14_len, vec23_14_w, vec23_14_ry = cls.extract_box_info(vec23_14, vec23_14_norm,
                                                                                 corners,
                                                                                 mid_point=mid_pts[:, :, 1])

        mask34_12 = tf.reshape(vec34_12_norm > vec23_14_norm, (-1, 1))
        mask_t = tf.cast(mask34_12, tf.float32)
        mask_n = tf.cast((~mask34_12), tf.float32)
        centroid = c_vec34_12 * mask_t + c_vec23_14 * mask_n
        length = vec34_12_len * mask_t + vec23_14_len * mask_n
        width = vec34_12_w * mask_t + vec23_14_w * mask_n
        ry = vec34_12_ry * mask_t + vec23_14_ry * mask_n

        a, b, c, d = tf.cast(gp, tf.float32)
        h1, h2 = tf.transpose(boxes_4cp[None, :, 8:10])
        cenx, cenz = tf.transpose(centroid[None, :, 0:2])

        y = -(a * cenx + c * cenz + d) / b
        ceny = y - h1
        height = h2 - h1
        box = tf.squeeze(tf.stack([cenx, ceny, cenz, length, width, height, ry], axis=1))

        return box

    @classmethod
    def convert_box_to_4cp_format(cls, bxs_3d, gp):
        if len(bxs_3d) == 0:
            return np.array([])
        else:
            anchors = cls.convert_box_to_axis_al_format(bxs_3d, True)

        cx, cy, cz, dx, dy, dz = tf.unstack(anchors, axis=1)

        # corner points
        xcs = tf.stack([dx / 2] * 2 + [-dx / 2] * 2, axis=1)
        zcs = tf.stack([dz / 2] + [-dz / 2] * 2 + [dz / 2], axis=1)

        rts = bxs_3d[:, 6]
        # find nearest multiple of pi/2
        half_pi = np.pi / 2
        ortho_rts = tf.round(rts / half_pi) * half_pi
        # find the angle with the nearest pi/2 multiple
        rts_dif = rts - ortho_rts
        zeros = tf.zeros_like(rts_dif, dtype=tf.float32)
        ones = tf.ones_like(rts_dif, dtype=tf.float32)
        # rotate and translate
        tr_matrix = tf.stack([tf.stack([tf.math.cos(rts_dif), tf.math.sin(rts_dif), cx], axis=1),
                              tf.stack([-tf.math.sin(rts_dif), tf.math.cos(rts_dif), cz], axis=1),
                              tf.stack([zeros, zeros, ones], axis=1)], axis=2)
        pts = tf.stack([xcs, zcs, tf.ones_like(xcs)], axis=1)
        crs = tf.reshape(tf.matmul(tf.transpose(tr_matrix, perm=[0, 2, 1]), pts)[:, :2], (-1, 8))
        # ground plane coefficients
        a, b, c, d = tf.cast(gp, tf.float32)

        y = -(a * cx + c * cz + d) / b
        h1 = (y - cy)
        h2 = (h1 + dy)
        # noinspection PyArgumentList
        return tf.concat([crs, tf.reshape(h1, (-1, 1)), tf.reshape(h2, (-1, 1))], axis=1)

    @staticmethod
    def convert_orientation_to_angle(or_v):
        return tf.stack([tf.math.cos(or_v), tf.math.sin(or_v)], axis=1)

    @staticmethod
    def convert_angle_to_orientation(a_vec):
        return tf.math.atan2(a_vec[:, 1], a_vec[:, 0])

    @staticmethod
    def extract_box_info(vec, vec_norm, points, mid_point):
        normalized_vec = vec / tf.reshape(vec_norm, (-1, 1))
        mid_pts = points - mid_point[:, :, None]

        lens = tf.transpose(tf.reduce_sum(tf.multiply(tf.transpose(mid_pts, perm=[2, 0, 1]), normalized_vec), axis=2))
        min_len = tf.math.reduce_min(lens, axis=1, keepdims=True)
        max_len = tf.math.reduce_max(lens, axis=1, keepdims=True)
        len_diff = max_len - min_len
        ortho_normalized_vec = tf.stack([-normalized_vec[:, 1], normalized_vec[:, 0]], axis=1)
        widths = tf.transpose(tf.reduce_sum(tf.multiply(tf.transpose(mid_pts, perm=[2, 0, 1]), ortho_normalized_vec),
                                            axis=2))
        min_w = tf.math.reduce_min(widths, axis=1, keepdims=True)
        max_w = tf.math.reduce_max(widths, axis=1, keepdims=True)
        width_diff = tf.reshape((max_w + min_w), (-1, 1))
        w_diff = tf.reshape(max_w - min_w, (-1, 1))
        ry = tf.reshape(-tf.atan2(vec[:, 1], vec[:, 0]), (-1, 1))
        cent = mid_point + normalized_vec * (min_len + max_len) / 2 + ortho_normalized_vec * width_diff
        return cent, len_diff, w_diff, ry

    @staticmethod
    def extract_corners_from_label(label_box: T) -> T:
        x, y, z, l, w, h, ry = label_box.T

        if len(label_box.shape) > 1 and label_box.shape[-1] != 1:
            c_x = torch.stack(2 * [l / 2] + 2 * [-l / 2] + 2 * [l / 2] + 2 * [-l / 2])
            c_y = torch.stack(4 * [torch.zeros(h.shape[0], device=label_box.device)] + 4 * [-h])
            c_z = torch.stack([w / 2] + 2 * [-w / 2] + 2 * [w / 2] + 2 * [-w / 2] + [w / 2])
            rotation_mat = create_rotation_matrix(angle=ry, dim=3, rotation='y')
            corners = multiply(rotation_mat, torch.stack([c_x, c_y, c_z]))
            corners = torch.stack(torch.split(corners, 8, dim=1)) + torch.stack([x, y, z]).T[..., None]
        else:
            c_x = np.array(2 * [l / 2] + 2 * [-l / 2] + 2 * [l / 2] + 2 * [-l / 2])
            c_y = np.array(4 * [0] + 4 * [-h])
            c_z = np.array([w / 2] + 2 * [-w / 2] + 2 * [w / 2] + 2 * [-w / 2] + [w / 2])
            rotation_mat = create_rotation_matrix(angle=ry, dim=3, rotation='y')
            corners = np.dot(rotation_mat, np.array([c_x, c_y, c_z]))
            corners[[0, 1, 2], :] += np.array([x, y, z])[:, None]
        return corners

    @staticmethod
    def extract_corners_from_axis_aligned(axis_aligned_boxes):
        x, y, z = axis_aligned_boxes[:, [0, 1, 2]].T
        dim_x, dim_y, dim_z = axis_aligned_boxes[:, [3, 4, 5]].T
        x_r = x + dim_x / 2
        x_l = x - dim_x / 2
        x_c = torch.stack(([x_r] * 2 + [x_l] * 2) * 2)
        y_d = y - dim_y
        y_c = torch.stack([y] * 4 + [y_d] * 4)
        z_r = z + dim_z / 2
        z_l = z - dim_z / 2
        z_c = torch.stack([z_r] + [z_l] * 2 + [z_r] * 2 + [z_l] * 2 + [z_r])

        corners_t = torch.cat([i.T.reshape(1, -1) for i in [x_c, y_c, z_c]])
        return corners_t

    @classmethod
    def project_box_to_img(cls, box, calib_dict, image_shape):
        device = box.device
        corners = cls.extract_corners_from_label(box)
        c_img = cls.project_lidar_to_img(corners, calib_dict=calib_dict)
        box_img = cls.box_2d_from_corners(c_img)
        h, w, _ = image_shape
        # boxes that are not in the image space i.e. they reside completely outside of it
        idxs_to_rm = (box_img[:, 2] < 0) | (box_img[:, 0] > w) | (box_img[:, 1] > h) | (box_img[:, 3] < 0)
        # remove boxes that have a height or width bigger than the 80% of image's height or width respectively
        h_box, w_box = (box_img[:, [3, 2]] - box_img[:, [1, 0]]).T
        mask = ((w_box > w * 0.8) | (h_box > h * 0.8))
        idxs_to_rm |= mask
        box_img = box_img[~idxs_to_rm]
        if len(box_img) > 0:
            # clip boxes to reside inside the image space
            box_img[:, 0] = torch.maximum(box_img[:, 0], torch.tensor([0], device=device)[:, None])
            box_img[:, 1] = torch.maximum(box_img[:, 1], torch.tensor([0], device=device)[:, None])
            box_img[:, 2] = torch.minimum(box_img[:, 2], torch.tensor([w], device=box.device)[:, None])
            box_img[:, 3] = torch.minimum(box_img[:, 3], torch.tensor([h], device=device)[:, None])
        return box_img, idxs_to_rm

    @staticmethod
    def project_lidar_to_img(pts, calib_dict):
        """
        pts: [N, 3, Y] or [3,Y]
        """
        camera_matrix = calib_dict['frame_calib_matrix'][2]
        camera_matrix = camera_matrix if tf.is_tensor(camera_matrix) else tf.convert_to_tensor(camera_matrix)
        shape = list(pts.shape)
        shape[-2] += 1
        if len(pts.shape) > 2:
            padded_pts[:, :3] = pts
        else:
            padded_pts = tf.concat([pts, tf.ones((1, shape[-1]))], axis=0)
        projected_pts = tf.matmul(camera_matrix, padded_pts)
        if len(projected_pts.shape) > 2:
            projected_pts = tf.transpose(projected_pts, (1, 0, 2))

        return tf.transpose(projected_pts[:2] / projected_pts[2])

    @classmethod
    def project_axis_aligned_to_img(cls, axis_aligned_anchors, image_shape, calib_dict=None, normalize=False,
                                    add_zero_col=False,
                                    tl_br_format=False):
        x, y, z, dim_x, dim_y, dim_z = tf.unstack(axis_aligned_anchors, axis=1)
        x_r = x + dim_x / 2.
        x_l = x - dim_x / 2.
        x_c = tf.stack(([x_r] * 2 + [x_l] * 2) * 2)
        y_d = y - dim_y
        y_c = tf.stack([y] * 4 + [y_d] * 4)
        z_r = z + dim_z / 2.
        z_l = z - dim_z / 2.
        z_c = tf.stack([z_r] + [z_l] * 2 + [z_r] * 2 + [z_l] * 2 + [z_r])
        corners_t = tf.concat([tf.reshape(tf.transpose(i), shape=(1, -1)) for i in [x_c, y_c, z_c]], axis=0)
        projected_points = cls.project_lidar_to_img(corners_t, calib_dict)
        x_min, y_min = tf.math.reduce_min(tf.reshape(tf.transpose(projected_points), (2, -1, 8)), axis=2)
        x_max, y_max = tf.math.reduce_max(tf.reshape(tf.transpose(projected_points), (2, -1, 8)), axis=2)

        anchors_img_space = tf.transpose(tf.stack([x_min, y_min, x_max, y_max]))
        image_shape = tf.cast(tf.squeeze(image_shape), dtype=tf.float32)
        norm_tensor = tf.stack([image_shape[..., 1], image_shape[..., 0]] * 2, axis=0)
        if tl_br_format:
            # anchors are returned in the format x1,y1,x2,y2 where (x1,y1) is the top left corner of the box and (x2,y2)
            # is the bottom right one
            anchors_img_space = tf.transpose(tf.stack([x_min, y_max, x_max, y_min]))
        if normalize:
            anchors_img_space = anchors_img_space / norm_tensor
        if add_zero_col:
            # adding image index inside batch.Since our batch size is 1 we add a zero column.will be used to roi_pool
            # noinspection PyArgumentList
            anchors_img_space = tf.concat(
                [tf.zeros((len(anchors_img_space), 1)), anchors_img_space],
                axis=1)

        return anchors_img_space

    @staticmethod
    def project_to_bev(axis_aligned_anchors: T, area_extents: T, normalize=False, **kwargs) -> T:
        """
        projects axis aligned boxes to bev space: [x, y, z, dim_x, dim_y, dim_z] -> [x1, y1, x2, y2]
        with the origin being the top left corner
        """
        add_zero_col = kwargs.get('add_zero_col', False)
        # calculate corners
        axa0, _, axa2, axa3, _, axa5 = tf.unstack(axis_aligned_anchors, axis=1)
        i0, i3 = axa0 - axa3 / 2, axa2 - axa5 / 2
        i2, i1 = axa0 + axa3 / 2, axa2 + axa5 / 2

        # move origin to the top left corner (flip z)
        i1, i3 = area_extents[-1, -1] - i1, area_extents[-1, -1] - i3
        i0, i2 = i0 - area_extents[0, 0], i2 - area_extents[0, 0]
        extent_rs = np.diff(area_extents).squeeze()
        extents = [extent_rs[0], extent_rs[2]] * 2

        anchors_bev = tf.stack([i0, i1, i2, i3], axis=1)
        if normalize:
            return anchors_bev / tf.convert_to_tensor(np.array(extents), dtype=tf.float32)
        return anchors_bev

    @classmethod
    def calculate_iou(cls, box1, box2):
        return cls.box_iou(box1, box2)

    @staticmethod
    def create_one_hot(cls_masked, neg_val=0.001, num_classes=2):
        pos_value = 1. - neg_val
        return tf.one_hot(cls_masked, num_classes, pos_value, neg_val)

    @staticmethod
    def box_iou(boxes1, boxes2):
        # torchvision.ops.boxes.box_iou
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        left_up = tf.maximum(boxes1[:, None, :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[:, None, 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area[:, None] + boxes2_area - inter_area

        return 1.0 * inter_area / union_area


# noinspection PyUnresolvedReferences
class BoxUtilsArray(BoxUtilsAbstract):

    @staticmethod
    def box_2d_from_corners(corners):
        """

        Parameters
        ----------
        corners: an array of 8x2 arrays i.e 8x2xN where N refers to the number of different boxes' corners

        Returns
        -------
        an array containing the two corners of the extracted 2d boxes in the following format:
        [xmin,ymin,xmax,ymax]
        """
        # transpose to correctly extract; for number of boxes>1
        xmin, ymin = np.amin(corners, axis=0).T
        xmax, ymax = np.amax(corners, axis=0).T
        # transpose for the case boxes>1 and reshape when boxes==1
        return np.array([xmin, ymin, xmax, ymax]).T.reshape(-1, 4)

    @staticmethod
    def offset_predictions_to_anchors(predictions, anchors):
        pred_anchors = np.zeros((len(predictions), 6), dtype=np.float32)
        pred_anchors[:, 0:3] = predictions[:, 0:3] * anchors[:, 3:] + anchors[:, 0:3]
        pred_anchors[:, 3:] = np.exp(np.log(anchors[:, 3:]) + predictions[:, 3:])
        return pred_anchors

    @classmethod
    def proposals_to_4c_box_al(cls, proposals, offset_scores, ground_plane):
        proposals_box_format = cls.convert_anchors_to_box_format(proposals, True)
        proposals_4cp_format = cls.convert_box_to_4cp_format(proposals_box_format, ground_plane)
        # test convert_4cp_to_box_format

        predictions_4c = convert_offsets_to_4cp_format(proposals_4cp_format, offset_scores)
        predictions_box = cls.convert_4cp_to_box_format(predictions_4c, ground_plane)
        pred_anchors = cls.convert_box_to_axis_al_format(predictions_box, True)

        return predictions_4c, predictions_box, pred_anchors, proposals_4cp_format

    @staticmethod
    def convert_box_to_axis_al_format(boxes_3d, ortho_rotate=False):
        anchors_axis_al = np.zeros((boxes_3d.shape[0], boxes_3d.shape[1] - 1))
        anchors_axis_al[:, [0, 1, 2, 4]] = boxes_3d[:, [0, 1, 2, 5]]
        # h stays the same i.e h = dim_y
        l, w, ry = boxes_3d[:, [3, 4, 6]].T
        if ortho_rotate:
            half_pi = np.pi / 2
            ry = np.round(ry / half_pi) * half_pi

        cos_ry, sin_ry = np.abs([np.cos(ry), np.sin(ry)])
        anchors_axis_al[:, 3] = l * cos_ry + w * sin_ry
        anchors_axis_al[:, 5] = w * cos_ry + l * sin_ry

        return anchors_axis_al

    @staticmethod
    def convert_anchors_to_box_format(anchors, get_longest_dim_as_len=False):
        boxes = np.zeros((len(anchors), 7), dtype=np.float32)
        boxes[:, 0:3] = anchors[:, 0:3]
        boxes[:, [3, 5, 4]] = anchors[:, 3:]
        if get_longest_dim_as_len:
            indices_to_swap = boxes[:, 4] > boxes[:, 3]
            # noinspection PyArgumentList
            boxes[indices_to_swap, 3:5] = boxes[indices_to_swap, [4, 3]]
            boxes[indices_to_swap, 6] = -np.pi / 2
        return boxes

    @staticmethod
    def convert_4cp_to_box_format(boxes_4cp, gp):
        # extracting corners
        corners = boxes_4cp[:, :8].reshape(-1, 2, 4)
        mid_pts = (corners[:, :, [0, 1, 2, 0]] + corners[:, :, [1, 2, 3, 3]]) / 2

        vec34_12 = mid_pts[:, :, 0] - mid_pts[:, :, 2]
        vec34_12_norm = np.linalg.norm(vec34_12, axis=1)

        vec23_14 = mid_pts[:, :, 3] - mid_pts[:, :, 1]
        vec23_14_norm = np.linalg.norm(vec23_14, axis=1)

        c_vec34_12, vec34_12_len, vec34_12_w, vec34_12_ry = BoxUtilsArray.extract_box_info(vec34_12, vec34_12_norm,
                                                                                           corners,
                                                                                           mid_point=mid_pts[:, :, 2])

        c_vec23_14, vec23_14_len, vec23_14_w, vec23_14_ry = BoxUtilsArray.extract_box_info(vec23_14, vec23_14_norm,
                                                                                           corners,
                                                                                           mid_point=mid_pts[:, :, 1])

        mask34_12 = vec34_12_norm > vec23_14_norm

        centroid = c_vec34_12 * mask34_12.astype(np.float32).reshape(-1, 1) + c_vec23_14 * (~mask34_12).astype(
            np.float32).reshape(-1, 1)
        length = vec34_12_len * mask34_12.astype(np.float32).reshape(-1, 1) + vec23_14_len * (~mask34_12).astype(
            np.float32).reshape(-1, 1)
        width = vec34_12_w * mask34_12.astype(np.float32).reshape(-1, 1) + vec23_14_w * (~mask34_12).astype(
            np.float32).reshape(-1, 1)
        ry = vec34_12_ry * mask34_12.astype(np.float32).reshape(-1, 1) + vec23_14_ry * (~mask34_12).astype(
            np.float32).reshape(-1, 1)

        a, b, c, d = gp
        h1, h2 = boxes_4cp[None, :, [8, 9]].T
        cenx, cenz = centroid[None, :, [0, 1]].T

        y = -(a * cenx + c * cenz + d) / b
        ceny = y - h1
        height = h2 - h1
        box = np.stack([cenx, ceny, cenz, length, width, height, ry], axis=1).squeeze()

        return box

    @classmethod
    def convert_box_to_4cp_format(cls, bxs_3d, gp):
        if len(bxs_3d) == 0:
            return np.array([])
        else:
            # bxs_3d = np.array([bxs_3d, bxs_3d]).squeeze(1)
            anchors = cls.convert_box_to_axis_al_format(bxs_3d, True)

        cx, cy, cz = anchors[:, [0, 1, 2]].T
        dx, dy, dz = anchors[:, [3, 4, 5]].T

        xcs = np.stack([dx / 2] * 2 + [-dx / 2] * 2, axis=1)
        zcs = np.stack([dz / 2] + [-dz / 2] * 2 + [dz / 2], axis=1)

        rts = bxs_3d[:, 6]

        half_pi = np.pi / 2
        ortho_rts = np.round(rts / half_pi) * half_pi

        rts_dif = rts - ortho_rts
        zeros = np.zeros_like(rts_dif)
        ones = np.ones_like(rts_dif)

        tr_matrix = np.stack([np.stack([np.cos(rts_dif), np.sin(rts_dif), cx], axis=1),
                              np.stack([-np.sin(rts_dif), np.cos(rts_dif), cz], axis=1),
                              np.stack([zeros, zeros, ones], axis=1)],
                             axis=2)
        pts = np.stack([xcs, zcs, np.ones_like(xcs)], axis=1)
        crs = np.matmul(np.transpose(tr_matrix, axes=[0, 2, 1]), pts)[:, :2].reshape(-1, 8)
        a, b, c, d = gp

        ground_y = -(a * cx + c * cz + d) / b
        h1 = (ground_y - cy)
        h2 = (h1 + dy)
        # return torch.as_tensor(np.concatenate([crs, h1.reshape(-1, 1), h2.reshape(-1, 1)], axis=1))
        return np.concatenate([crs, h1.reshape(-1, 1), h2.reshape(-1, 1)], axis=1)

    @staticmethod
    def convert_orientation_to_angle(or_v):
        return np.stack([np.cos(or_v), np.sin(or_v)], axis=1)

    @staticmethod
    def convert_angle_to_orientation(a_vec):
        return np.arctan2(a_vec[:, 1], a_vec[:, 0])

    @staticmethod
    def extract_box_info(vec, vec_norm, points, mid_point):
        normalized_vec = vec / vec_norm.reshape(-1, 1)
        mid_pts = points - mid_point[:, :, None]
        # mid_pts = points - mid_point
        lens = np.sum(np.multiply(np.transpose(mid_pts, axes=[2, 0, 1]), normalized_vec), axis=2).T
        min_len = np.min(lens, axis=1, keepdims=True)
        max_len = np.max(lens, axis=1, keepdims=True)
        len_diff = max_len - min_len
        ortho_normalized_vec = np.stack([-normalized_vec[:, 1], normalized_vec[:, 0]], axis=1)
        widths = np.sum(np.multiply(np.transpose(mid_pts, axes=[2, 0, 1]), ortho_normalized_vec), axis=2).T
        min_w = np.min(widths, axis=1, keepdims=True)
        max_w = np.max(widths, axis=1, keepdims=True)
        width_diff = (max_w + min_w).reshape(-1, 1)
        w_diff = (max_w - min_w).reshape(-1, 1)
        ry = -np.arctan2(vec[:, 1], vec[:, 0]).reshape(-1, 1)
        cent = mid_point + normalized_vec * (min_len + max_len) / 2 + ortho_normalized_vec * width_diff
        return cent, len_diff, w_diff, ry

    @staticmethod
    def extract_corners_from_label(label_box: T) -> T:
        x, y, z, l, w, h, ry = label_box.T

        if len(label_box.shape) > 1 and label_box.shape[-1] != 1:
            c_x = np.array(2 * [l / 2] + 2 * [-l / 2] + 2 * [l / 2] + 2 * [-l / 2])

            c_y = np.array(4 * [np.zeros(h.shape[0])] + 4 * [-h])

            c_z = np.array([w / 2] + 2 * [-w / 2] + 2 * [w / 2] + 2 * [-w / 2] + [w / 2])
            rotation_mat = create_rotation_matrix(angle=ry, dim=3, rotation='y')

            corners = multiply(rotation_mat, np.array([c_x, c_y, c_z]))
            corners = np.array(np.split(corners, corners.shape[-1] // 8, axis=1)) + np.array([x, y, z]).T[..., None]
        else:
            c_x = np.array(2 * [l / 2] + 2 * [-l / 2] + 2 * [l / 2] + 2 * [-l / 2])

            c_y = np.array(4 * [0] + 4 * [-h])

            c_z = np.array([w / 2] + 2 * [-w / 2] + 2 * [w / 2] + 2 * [-w / 2] + [w / 2])
            rotation_mat = create_rotation_matrix(angle=ry, dim=3, rotation='y')

            corners = np.dot(rotation_mat, np.array([c_x, c_y, c_z]))
            corners[[0, 1, 2], :] += np.array([x, y, z])[:, None]

        return corners

    @staticmethod
    def extract_corners_from_axis_aligned(axis_aligned_boxes):
        x, y, z = axis_aligned_boxes[:, [0, 1, 2]].T
        dim_x, dim_y, dim_z = axis_aligned_boxes[:, [3, 4, 5]].T

        x_r = x + dim_x / 2
        x_l = x - dim_x / 2
        x_c = np.array(([x_r] * 2 + [x_l] * 2) * 2).T

        y_d = y - dim_y
        y_c = np.array([y] * 4 + [y_d] * 4).T

        z_r = z + dim_z / 2
        z_l = z - dim_z / 2
        z_c = np.array([z_r] + [z_l] * 2 + [z_r] * 2 + [z_l] * 2 + [z_r]).T

        corners = np.vstack([i.reshape(1, -1) for i in [x_c, y_c, z_c]])
        return corners

    @classmethod
    def project_box_to_img(cls, box, calib_dict, image_shape):
        corners = cls.extract_corners_from_label(box)
        c_img = cls.project_lidar_to_img(corners, calib_dict=calib_dict)
        box_img = cls.box_2d_from_corners(c_img)
        h, w, _ = image_shape
        # boxes that are not in the image space i.e. they reside completely outside of it
        idxs_to_rm = (box_img[:, 2] < 0) | (box_img[:, 0] > w) | (box_img[:, 1] > h) | (box_img[:, 3] < 0)
        # remove boxes that have a height or width bigger than the 80% of image's height or width respectively
        h_box, w_box = (box_img[:, [3, 2]] - box_img[:, [1, 0]]).T
        mask = ((w_box > w * 0.8) | (h_box > h * 0.8))
        idxs_to_rm |= mask
        box_img = box_img[~idxs_to_rm]
        if len(box_img) > 0:
            # clip boxes to reside inside the image space
            box_img[:, 0] = np.maximum(box_img[:, 0], np.array([0])[:, None])
            box_img[:, 1] = np.maximum(box_img[:, 1], np.array([0])[:, None])
            box_img[:, 2] = np.minimum(box_img[:, 2], np.array([w])[:, None])
            box_img[:, 3] = np.minimum(box_img[:, 3], np.array([h])[:, None])
        return box_img, idxs_to_rm

    @staticmethod
    def project_lidar_to_img(pts, calib_dict=None):

        camera_matrix = calib_dict['frame_calib_matrix'][2]
        shape = list(pts.shape)
        shape[-2] += 1
        padded_pts = np.ones(shape)
        if len(pts.shape) > 2:
            padded_pts[:, :3] = pts
        else:
            padded_pts[:3] = pts
        pts_2d = np.matmul(camera_matrix, padded_pts)
        if len(pts_2d.shape) > 2:
            pts_2d = pts_2d.transpose(1, 0, 2)
        pts_2d[[0, 1], :] /= pts_2d[[2, 2], :]

        return pts_2d[:2].T

    @classmethod
    def project_axis_aligned_to_img(cls, axis_aligned_anchors, image_shape, calib_dict=None, normalize=False):
        '''

        :return:anchors to image space
        '''
        x, y, z = axis_aligned_anchors[:, [0, 1, 2]].T
        dim_x, dim_y, dim_z = axis_aligned_anchors[:, [3, 4, 5]].T

        x_r = x + dim_x / 2.
        x_l = x - dim_x / 2.
        x_c = np.array(([x_r] * 2 + [x_l] * 2) * 2).T

        y_d = y - dim_y
        y_c = np.array([y] * 4 + [y_d] * 4).T

        z_r = z + dim_z / 2.
        z_l = z - dim_z / 2.
        z_c = np.array([z_r] + [z_l] * 2 + [z_r] * 2 + [z_l] * 2 + [z_r]).T

        corners = np.vstack([i.reshape(1, -1) for i in [x_c, y_c, z_c]])
        projected_points = cls.project_lidar_to_img(corners, calib_dict)
        x_min, y_min = np.amin(projected_points.T.reshape(2, -1, 8), axis=2)
        x_max, y_max = np.amax(projected_points.T.reshape(2, -1, 8), axis=2)

        anchors_img_space = np.stack([x_min, y_min, x_max, y_max]).T
        if normalize:
            return anchors_img_space / np.array([image_shape[1], image_shape[0]] * 2)
        return anchors_img_space

    @staticmethod
    def project_to_bev(axis_aligned_anchors: T, area_extents: T, normalize: bool = False, **kwargs) -> T:
        anchors_bev = np.zeros((len(axis_aligned_anchors), 4), dtype=np.float32)
        anchors_bev[:, [0, 3]] = axis_aligned_anchors[:, [0, 2]] - axis_aligned_anchors[:, [3, 5]] / 2
        anchors_bev[:, [2, 1]] = axis_aligned_anchors[:, [0, 2]] + axis_aligned_anchors[:, [3, 5]] / 2
        anchors_bev[:, [1, 3]] = area_extents[-1, -1] - anchors_bev[:, [1, 3]]
        anchors_bev[:, [0, 2]] -= area_extents[0, 0]
        if normalize:
            extent_rs = np.diff(area_extents).squeeze()
            return anchors_bev / np.array([extent_rs[0], extent_rs[2]] * 2)
        return anchors_bev

    @staticmethod
    def calculate_iou(box1, box2):
        inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
        inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
        inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
        inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])
        inter_h = inter_y2 - inter_y1
        inter_w = inter_x2 - inter_x1
        inter_area = inter_h * inter_w
        inter_area[(inter_h <= 0) | (inter_w <= 0)] = 0
        union_area = np.multiply((box1[:, 2] - box1[:, 0]), (box1[:, 3] - box1[:, 1]))[:, None] + np.multiply(
            (box2[:, 2] - box2[:, 0]), (
                    box2[:, 3] - box2[:, 1])) - inter_area
        iou = inter_area / (union_area + 1e-16)
        iou[(inter_h <= 0) | (inter_w <= 0)] = 0
        return iou.round(3)

    @staticmethod
    def create_one_hot(cls_masked, neg_val=0.001, num_classes=2):
        one_hot_vec = np.zeros((num_classes, num_classes), dtype=np.float32) + neg_val
        one_hot_vec[range(num_classes), range(num_classes)] = 1. - neg_val
        return one_hot_vec[cls_masked.astype(np.uint64)]


def call_proper_method(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        name = method.__name__
        if tf.is_tensor(args[0]):
            fn = getattr(BoxUtilsTensor, name)
        elif isinstance(args[0], np.ndarray):
            fn = getattr(BoxUtilsArray, name)
        else:
            raise Exception('argument must be of type tf.Tensor or numpy.ndarray')
        return fn(*args, **kwargs)

    return wrapper


@call_proper_method
def box_2d_from_corners(corners): ...


@call_proper_method
def offset_predictions_to_anchors(predictions, anchors): ...


# test convert_4cp_to_box_format
@call_proper_method
def proposals_to_4c_box_al(proposals, offset_scores, ground_plane): ...


@call_proper_method
def convert_box_to_axis_al_format(anchors_3d, ortho_rotate=False): ...


@call_proper_method
def convert_anchors_to_box_format(): ...


@call_proper_method
def convert_4cp_to_box_format(boxes_4cp, gp): ...


@call_proper_method
def convert_box_to_4cp_format(bxs_3d, gp): ...


@call_proper_method
def convert_orientation_to_angle(or_v):
    """
    [θ] -> [cos(θ), sin(θ)]
    """
    ...


@call_proper_method
def convert_angle_to_orientation(a_vec):
    """
    https://en.wikipedia.org/wiki/Atan2
    [x,y]-> atan2(y,x) -> θ
    """
    ...


@call_proper_method
def extract_box_info(vec, vec_norm, points, mid_point): ...


@call_proper_method
def extract_corners_from_label(label_box: T) -> T: ...


@call_proper_method
def project_box_to_img(box, calib_dict, image_shape): ...


@call_proper_method
def project_lidar_to_img(pts, calib_dict=None):
    """
    https://en.wikipedia.org/wiki/Camera_matrix#Derivation
    https://en.wikipedia.org/wiki/3D_projection#Perspective_projection
    https://www.iitr.ac.in/departments/MA/uploads/Unit%202%20lec-1.pdf
    """
    ...


@call_proper_method
def project_axis_aligned_to_img(axis_aligned_anchors, image_shape, calib_dict=None, normalize=False):
    ...


@call_proper_method
def project_to_bev(axis_aligned_anchors: T, area_extents: T, normalize=False, **kwargs) -> T: ...


@call_proper_method
def calculate_iou(box1, box2): ...


def convert_4cp_to_offsets(boxes_4c, gt_boxes_4c):
    return gt_boxes_4c - boxes_4c


def convert_offsets_to_4cp_format(boxes_4c, offsets):
    return boxes_4c + offsets
