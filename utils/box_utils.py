import torch
import numpy as np
from typing import Union
from abc import ABC, abstractmethod
from functools import wraps
from torchvision.ops.boxes import box_iou
from utils.general_utils import multiply, create_rotation_matrix

T = Union[torch.Tensor, np.ndarray]


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
    @torch.jit.script
    def offset_predictions_to_anchors(predictions, anchors):
        pred_anchors = torch.zeros((len(predictions), 6), device=anchors.device)
        pred_anchors[:, 0:3] = predictions[:, 0:3] * anchors[:, 3:] + anchors[:, 0:3]
        pred_anchors[:, 3:] = torch.exp(torch.log(anchors[:, 3:]) + predictions[:, 3:])
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
        """
        converts box to axis aligned format:
        [x, y, z, l, w, h, ry] -> [x, y, z, dim_x, dim_y, dim_z]
        """
        anchors_axis_al = torch.zeros((boxes_3d.shape[0], boxes_3d.shape[1] - 1), device=boxes_3d.device)
        # x, y, z, dy
        anchors_axis_al[:, [0, 1, 2, 4]] = boxes_3d[:, [0, 1, 2, 5]]
        # h stays the same i.e h = dim_y
        # calculate dx, dz given l, w, ry
        l, w, ry = boxes_3d[:, [3, 4, 6]].T
        if ortho_rotate:
            half_pi = np.pi / 2
            ry = torch.round(ry / half_pi) * half_pi

        cos_ry = torch.abs(torch.cos(ry))
        sin_ry = torch.abs(torch.sin(ry))
        # dx
        anchors_axis_al[:, 3] = l * cos_ry + w * sin_ry
        # dz
        anchors_axis_al[:, 5] = w * cos_ry + l * sin_ry

        return anchors_axis_al

    @staticmethod
    def convert_anchors_to_box_format(anchors, get_longest_dim_as_len=False):
        boxes = torch.zeros((len(anchors), 7), dtype=torch.float32, device=anchors.device)
        boxes[:, 0:3] = anchors[:, 0:3]
        boxes[:, [3, 5, 4]] = anchors[:, 3:]
        if get_longest_dim_as_len:
            indices_to_swap = boxes[:, 4] > boxes[:, 3]
            # noinspection PyArgumentList
            boxes[indices_to_swap, 3:5] = torch.index_select(boxes[indices_to_swap], 1,
                                                             torch.tensor([4, 3], device=anchors.device))
            # to(anchors.device))
            boxes[indices_to_swap, 6] = -np.pi / 2
        return boxes

    @classmethod
    def convert_4cp_to_box_format(cls, boxes_4cp, gp):
        # extracting corners
        corners = boxes_4cp[:, :8].reshape(-1, 2, 4)
        mid_pts = (corners[:, :, [0, 1, 2, 0]] + corners[:, :, [1, 2, 3, 3]]) / 2

        vec34_12 = mid_pts[:, :, 0] - mid_pts[:, :, 2]
        vec34_12_norm = torch.linalg.norm(vec34_12, axis=1)

        vec23_14 = mid_pts[:, :, 3] - mid_pts[:, :, 1]
        vec23_14_norm = torch.linalg.norm(vec23_14, axis=1)

        c_vec34_12, vec34_12_len, vec34_12_w, vec34_12_ry = cls.extract_box_info(vec34_12, vec34_12_norm,
                                                                                 corners,
                                                                                 mid_point=mid_pts[:, :, 2])

        c_vec23_14, vec23_14_len, vec23_14_w, vec23_14_ry = cls.extract_box_info(vec23_14, vec23_14_norm,
                                                                                 corners,
                                                                                 mid_point=mid_pts[:, :, 1])

        mask34_12 = vec34_12_norm > vec23_14_norm

        centroid = c_vec34_12 * mask34_12.float().reshape(-1, 1) + c_vec23_14 * (~mask34_12).float().reshape(-1, 1)
        length = vec34_12_len * mask34_12.float().reshape(-1, 1) + vec23_14_len * (~mask34_12).float().reshape(-1, 1)
        width = vec34_12_w * mask34_12.float().reshape(-1, 1) + vec23_14_w * (~mask34_12).float().reshape(-1, 1)
        ry = vec34_12_ry * mask34_12.float().reshape(-1, 1) + vec23_14_ry * (~mask34_12).float().reshape(-1, 1)

        a, b, c, d = gp
        h1, h2 = boxes_4cp[None, :, [8, 9]].T
        cenx, cenz = centroid[None, :, [0, 1]].T

        y = -(a * cenx + c * cenz + d) / b
        ceny = y - h1
        height = h2 - h1
        box = torch.stack([cenx, ceny, cenz, length, width, height, ry], dim=1).squeeze()

        return box

    @classmethod
    def convert_box_to_4cp_format(cls, bxs_3d, gp):
        if len(bxs_3d) == 0:
            return np.array([])
        else:
            anchors = cls.convert_box_to_axis_al_format(bxs_3d, True)

        cx, cy, cz = anchors[:, [0, 1, 2]].T
        dx, dy, dz = anchors[:, [3, 4, 5]].T

        # corner points
        xcs = torch.stack([dx / 2] * 2 + [-dx / 2] * 2, dim=1)
        zcs = torch.stack([dz / 2] + [-dz / 2] * 2 + [dz / 2], dim=1)

        rts = bxs_3d[:, 6]
        # find nearest multiple of pi/2
        half_pi = np.pi / 2
        ortho_rts = torch.round(rts / half_pi) * half_pi
        # find the angle with the nearest pi/2 multiple
        rts_dif = rts - ortho_rts
        zeros = torch.zeros_like(rts_dif, dtype=torch.float32)
        ones = torch.ones_like(rts_dif, dtype=torch.float32)
        # rotate and translate
        tr_matrix = torch.stack([torch.stack([torch.cos(rts_dif), torch.sin(rts_dif), cx], dim=1),
                                 torch.stack([-torch.sin(rts_dif), torch.cos(rts_dif), cz], dim=1),
                                 torch.stack([zeros, zeros, ones], dim=1)],
                                dim=2)
        pts = torch.stack([xcs, zcs, torch.ones_like(xcs)], dim=1)
        crs = torch.matmul(torch.transpose(tr_matrix, 2, 1), pts)[:, :2].reshape(-1, 8)
        # ground plane coefficients
        a, b, c, d = gp

        y = -(a * cx + c * cz + d) / b
        h1 = (y - cy)
        h2 = (h1 + dy)
        # noinspection PyArgumentList
        return torch.cat([crs, h1.reshape(-1, 1), h2.reshape(-1, 1)], dim=1)

    @staticmethod
    @torch.jit.script
    def convert_orientation_to_angle(or_v):
        return torch.stack([torch.cos(or_v), torch.sin(or_v)], dim=1)

    @staticmethod
    def convert_angle_to_orientation(a_vec):
        return torch.atan2(a_vec[:, 1], a_vec[:, 0])

    @staticmethod
    def extract_box_info(vec, vec_norm, points, mid_point):
        normalized_vec = vec / vec_norm.reshape(-1, 1)
        mid_pts = points - mid_point[:, :, None]
        # mid_pts = points - mid_point

        lens = torch.sum(torch.multiply(mid_pts.permute([2, 0, 1]), normalized_vec), dim=2).T
        min_len, _ = torch.min(lens, dim=1, keepdim=True)
        max_len, _ = torch.max(lens, dim=1, keepdim=True)
        len_diff = max_len - min_len
        ortho_normalized_vec = torch.stack([-normalized_vec[:, 1], normalized_vec[:, 0]], dim=1)
        widths = torch.sum(torch.multiply(mid_pts.permute([2, 0, 1]), ortho_normalized_vec), dim=2).T
        min_w, _ = torch.min(widths, dim=1, keepdim=True)
        max_w, _ = torch.max(widths, dim=1, keepdim=True)
        width_diff = (max_w + min_w).reshape(-1, 1)
        w_diff = (max_w - min_w).reshape(-1, 1)
        ry = -torch.atan2(vec[:, 1], vec[:, 0]).reshape(-1, 1)
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
        # return box_img.reshape(-1) if box_img.shape[0] == 1 else box_img
        return box_img, idxs_to_rm

    @staticmethod
    def project_lidar_to_img(pts, calib_dict):
        """
        pts: [N, 3, Y] or [3,Y]
        """
        camera_matrix = calib_dict['frame_calib_matrix'][2]
        camera_matrix = camera_matrix if isinstance(camera_matrix, torch.Tensor) else torch.as_tensor(camera_matrix,
                                                                                                      device=pts.device)
        shape = list(pts.shape)
        shape[-2] += 1
        padded_pts = torch.ones(shape, device=pts.device)
        if len(pts.shape) > 2:
            padded_pts[:, :3] = pts
        else:
            padded_pts[:3] = pts
        projected_pts = torch.matmul(camera_matrix, padded_pts)
        if len(projected_pts.shape) > 2:
            projected_pts = projected_pts.permute(1, 0, 2)
        projected_pts[[0, 1], :] /= projected_pts[[2, 2], :]

        return projected_pts[:2].T

    @classmethod
    def project_axis_aligned_to_img(cls, axis_aligned_anchors, image_shape, calib_dict=None, normalize=False,
                                    add_zero_col=False,
                                    tl_br_format=False):
        x, y, z = axis_aligned_anchors[:, [0, 1, 2]].T
        dim_x, dim_y, dim_z = axis_aligned_anchors[:, [3, 4, 5]].T
        x_r = x + dim_x / 2.
        x_l = x - dim_x / 2.
        x_c = torch.stack(([x_r] * 2 + [x_l] * 2) * 2)
        y_d = y - dim_y
        y_c = torch.stack([y] * 4 + [y_d] * 4)
        z_r = z + dim_z / 2.
        z_l = z - dim_z / 2.
        z_c = torch.stack([z_r] + [z_l] * 2 + [z_r] * 2 + [z_l] * 2 + [z_r])

        corners_t = torch.cat([i.T.reshape(1, -1) for i in [x_c, y_c, z_c]])

        projected_points = cls.project_lidar_to_img(corners_t, calib_dict)

        x_min, y_min = torch.amin(projected_points.T.reshape(2, -1, 8), dim=2)
        x_max, y_max = torch.amax(projected_points.T.reshape(2, -1, 8), dim=2)

        anchors_img_space = torch.stack([x_min, y_min, x_max, y_max]).T
        image_shape = image_shape.squeeze()
        norm_tensor = torch.stack([image_shape[..., 1], image_shape[..., 0]] * 2, dim=0)
        if tl_br_format:
            # anchors are returned in the format x1,y1,x2,y2 where (x1,y1) is the top left corner of the box and (x2,y2)
            # is the bottom right one
            anchors_img_space = torch.stack([x_min, y_max, x_max, y_min]).T
        if normalize:
            anchors_img_space = anchors_img_space / norm_tensor
        if add_zero_col:
            # adding image index inside batch.Since our batch size is 1 we add a zero column.will be used to roi_pool
            # noinspection PyArgumentList
            anchors_img_space = torch.cat(
                [torch.zeros((len(anchors_img_space), 1), device=axis_aligned_anchors.device), anchors_img_space],
                dim=1)

        return anchors_img_space.float()

    @staticmethod
    def project_to_bev(axis_aligned_anchors: T, area_extents: T, normalize=False, **kwargs) -> T:
        """
        projects axis aligned boxes to bev space: [x, y, z, dim_x, dim_y, dim_z] -> [x1, y1, x2, y2]
        with the origin being the top left corner
        """
        add_zero_col = kwargs.get('add_zero_col', False)
        anchors_bev = torch.zeros((len(axis_aligned_anchors), 4), device=axis_aligned_anchors.device)
        # calculate corners
        anchors_bev[:, [0, 3]] = axis_aligned_anchors[:, [0, 2]] - axis_aligned_anchors[:, [3, 5]] / 2
        anchors_bev[:, [2, 1]] = axis_aligned_anchors[:, [0, 2]] + axis_aligned_anchors[:, [3, 5]] / 2
        # move origin to the top left corner (flip z)
        anchors_bev[:, [1, 3]] = area_extents[-1, -1] - anchors_bev[:, [1, 3]]
        anchors_bev[:, [0, 2]] -= area_extents[0, 0]
        extent_rs = np.diff(area_extents).squeeze()
        extents = [extent_rs[0], extent_rs[2]] * 2
        if add_zero_col:
            # add batch column for roi layer
            zeros = torch.zeros((anchors_bev.shape[0], anchors_bev.shape[1] + 1), device=axis_aligned_anchors.device)
            zeros[:, 1:] = anchors_bev
            anchors_bev = zeros
            # add one for dividing with the batch column
            extents = [1] + extents
        if normalize:
            return anchors_bev / torch.as_tensor(np.array(extents), device=axis_aligned_anchors.device)
        return anchors_bev

    @staticmethod
    def calculate_iou(box1, box2):
        return box_iou(box1, box2)

    @staticmethod
    def create_one_hot(cls_masked, neg_val=0.001, num_classes=2):
        one_hot_vec = torch.zeros(num_classes, num_classes).to(cls_masked.device) + neg_val
        one_hot_vec[range(num_classes), range(num_classes)] = 1. - neg_val
        return one_hot_vec[cls_masked.long()]


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
        functions = {np.ndarray: getattr(BoxUtilsArray, name), torch.Tensor: getattr(BoxUtilsTensor, name)}
        return functions[type(args[0])](*args, **kwargs)

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
