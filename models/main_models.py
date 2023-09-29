import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from configs.config import default_config
from general_utils import export
from torchvision.ops import nms, roi_pool, roi_align, RoIAlign, RoIPool
from utils.losses import SoftmaxCELoss, WeightedSmoothL1Loss
from .building_blocks import FeatureExtractor, Fusion, Predictor, ConvBlock
from utils.box_utils import convert_angle_to_orientation, project_axis_aligned_to_img, project_to_bev, \
    proposals_to_4c_box_al, offset_predictions_to_anchors

from data_utils.dataset_utils import create_mini_batch_mask, create_target_offsets, create_target_angle, create_one_hot, \
    create_mini_batch_mask_rpn

__all__ = []


class FeatureExtractors(nn.Module):
    _feature_extractors_config_ = 'feature_extractors'
    _img_input_channels_ = 'img_input_channels'
    _bev_input_channels_ = 'bev_input_channels'
    _output_channels_ = 'out_channels'
    _activation_function_ = 'activation_function'
    _batch_normalization_ = 'batch_normalization'

    def __init__(self, config=None, batch_norm=False):
        super(FeatureExtractors, self).__init__()
        config = config[self._feature_extractors_config_]

        image_input_channels = config[self._img_input_channels_]
        bev_input_channels = config[self._bev_input_channels_]
        output_channels = config[self._output_channels_]
        activation = config[self._activation_function_]
        enable_batch_norm = config[self._batch_normalization_]

        self.feature_extractor_img = FeatureExtractor(image_input_channels, output_channels,
                                                      batch_norm=enable_batch_norm,
                                                      activation=activation)
        self.feature_extractor_bev = FeatureExtractor(bev_input_channels, output_channels, batch_norm=enable_batch_norm,
                                                      activation=activation)

    def forward(self, x1, x2):
        return self.feature_extractor_img(x1), self.feature_extractor_bev(x2)[:, :, 4:, :]


@export
class RPN(nn.Module):
    _roi_type_dict = {'roi_align': RoIAlign, 'roi_pool': RoIPool}
    _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 2, 'spatial_scale': 1.},
                         'roi_pool': {'spatial_scale': 1.}}
    _batch_normalization_ = 'batch_normalization'

    _rpn_key_ = 'rpn'
    _input_channels_key_ = 'input_channels'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_size_key_ = 'nms_size'
    _predictors_key_ = 'predictors'
    _mini_batch_size_key_ = 'mini_batch_size'
    _regression_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'

    def __init__(self, config=None):
        super(RPN, self).__init__()

        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._rpn_key_)

        self.input_channels = config.get(self._input_channels_key_)
        self.roi_size = config.get(self._roi_size_key_, 3)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.8)
        self.nms_size = config.get(self._nms_size_key_, 1024)
        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)
        self.roi_layer_bev = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)
        enable_batch_norm = config.get(self._batch_normalization_)


        self.conv11_img = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=enable_batch_norm,
                                    activation='relu')

        self.conv11_bev = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=enable_batch_norm,
                                    activation='relu')

        self.fusion = Fusion()

        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.regression_weight = config.get(self._regression_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)

        self.regression_loss = WeightedSmoothL1Loss(self.regression_weight)
        self.classification_loss = SoftmaxCELoss(self.classification_weight)

    def forward(self, img_map, bev_map, anchors, mask=None, targets=None, img_mask=None, bev_mask=None):
        # the anchors(axis aligned format) that were calculated by the dataset
        anchors_img, anchors_bev, filtered_anchors = anchors

        #  feed RPN with the previously acquired feature maps
        # RPN FOLLOWS
        # first there is a 1x1 conv layer followed by a ROI Pooling layer
        x1 = self.conv11_img(img_map)
        x2 = self.conv11_bev(bev_map)
        # crop and resize operation
        # ROI Pooling/Align https://pytorch.org/vision/stable/ops.html
        # extract feature regions using the projected anchors
        # extracted regions are of 3x3 size
        # anchors_img, anchors_bev represent the filtered anchors
        # projected into image and bev spaces respectively
        # img_mask, bev_mask are either .0 or 1. and are used for the path drop functionality i.e. when one of them
        # is .0 the equivalent path gets dropped;only one of them can be .0
        # roi_rgb = self.roi_layer_img(img_mask * x1, anchors_img, output_size=self.roi_size, **self.roi_function_params)
        # roi_bev = self.roi_layer_bev(bev_mask * x2, anchors_bev, output_size=self.roi_size, **self.roi_function_params)
        roi_rgb = self.roi_layer_img(img_mask * x1, anchors_img)
        roi_bev = self.roi_layer_bev(bev_mask * x2, anchors_bev)

        # Fusion
        # the feature crops of the previous step are fused using an element-wise mean operation
        fused = self.fusion(roi_rgb, roi_bev, img_mask + bev_mask)
        # Fully connected layers
        # instead of fully connected layers convolutional ones can be used
        objectness, offsets = self.predictor(fused)
        # obtain the predicted anchors(using offsets and filtered anchors) and project them to bev space for nms
        reg_anchors = offset_predictions_to_anchors(offsets, filtered_anchors)

        bev_proposals = project_to_bev(reg_anchors, self.area_extents, normalize=False)

        # apply NMS to obtain the 1024 proposals with the higher score
        top_proposal_indices = nms(bev_proposals, F.softmax(objectness, dim=1)[:, 1],
                                   self.nms_threshold)[:self.nms_size]

        rpn_loss = None
        if targets is not None:
            gt_offsets, gt_objectness = targets

            rpn_loss = self.regression_loss(offsets[mask], gt_offsets[mask], objectness_gt=gt_objectness[mask]) + \
                       self.classification_loss(objectness[mask], gt_objectness[mask])

        return reg_anchors[top_proposal_indices], objectness[top_proposal_indices], rpn_loss


class SecondStageDetector(nn.Module):
    _roi_type_dict = {'roi_align': RoIAlign, 'roi_pool': RoIPool}
    # _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 6}, 'roi_pool': {}}
    _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 6, 'spatial_scale': 1.},
                         'roi_pool': {'spatial_scale': 1.}}
    _batch_normalization_ = 'batch_normalization'

    _second_stage_detector_key_ = 'second_stage_detector'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_size_key_ = 'nms_size'
    _mini_batch_size_key_ = 'mini_batch_size'
    _predictors_key_ = 'predictors'
    _localization_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'
    _orientation_weight_key_ = 'orientation_weight'

    def __init__(self, config=None):
        super(SecondStageDetector, self).__init__()

        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._second_stage_detector_key_)

        self.roi_size = config.get(self._roi_size_key_, 7)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.01)
        self.nms_size = config.get(self._nms_size_key_, 100)
        self.mini_batch_size = config.get(self._mini_batch_size_key_, 1024)
        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)
        self.roi_layer_bev = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)


        self.fusion = Fusion()
        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.localization_weight = config.get(self._localization_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)
        self.orientation_weight = config.get(self._orientation_weight_key_, 1.0)
        # losses
        self.classification_loss = SoftmaxCELoss(self.classification_weight)
        self.localization_loss = WeightedSmoothL1Loss(self.localization_weight)
        self.orientation_loss = WeightedSmoothL1Loss(self.orientation_weight)

    # noinspection DuplicatedCode
    def forward(self, img_feat_map, bev_feat_map, top_anchors, image_shape, calibration_dict,
                ground_plane, rpn_loss=None, targets=None, img_mask=None, bev_mask=None):
        bev_ins = project_to_bev(top_anchors, self.area_extents, add_zero_col=True)
        #
        rgb_ins = project_axis_aligned_to_img(top_anchors, image_shape, calibration_dict[0], normalize=False,
                                              add_zero_col=True)

        # ROI Pooling
        rois_rgb = self.roi_layer_img(img_mask * img_feat_map, rgb_ins)
        rois_bev = self.roi_layer_bev(bev_mask * bev_feat_map, bev_ins)
        # Fusion
        fused_avod = self.fusion(rois_rgb, rois_bev, img_mask + bev_mask)

        # Feed fully connected layers with feature crops
        obj_score, offset_scores, ang_scores = self.predictor(fused_avod)

        obj_score_softmax = F.softmax(obj_score, dim=1)

        orientations = convert_angle_to_orientation(ang_scores)

        predictions_4c, predictions_box, pred_anchors, proposals_4cp_format = proposals_to_4c_box_al(top_anchors,
                                                                                                     offset_scores,
                                                                                                     ground_plane)

        pred_bev = project_to_bev(pred_anchors, self.area_extents, normalize=False)
        top_pred_scores, _ = torch.max(obj_score[:, 1:], dim=1)
        top_indices = nms(pred_bev, top_pred_scores, self.nms_threshold)[:self.nms_size]

        # top_scores = obj_score[top_indices]
        top_scores_soft = obj_score_softmax[top_indices]
        top_pred_anchors = pred_anchors[top_indices]
        top_pred_orientations = orientations[top_indices]

        final_loss = None
        avod_loc_loss = None
        avod_cls_loss = None
        avod_angle_loss = None
        # if we are training the section below will be executed
        if rpn_loss:
            gt_anchors_bev, label_boxes, label_cls_ids = targets
            mb_mask, cls_masked, gt_masked = create_mini_batch_mask(gt_anchors_bev, bev_ins[:, 1:], label_cls_ids,
                                                                    size=self.mini_batch_size)
            # Create the target vectors

            # receives the masked proposals in 4cp format along with the labels and creates the target offsets
            # that will be used for computing the localization loss
            offsets_gt_masked = create_target_offsets(label_boxes[gt_masked], proposals_4cp_format[mb_mask],
                                                      ground_plane)

            gt_masked_cls = create_one_hot(cls_masked, neg_val=0.0010000000474974513)

            gt_angle_vec = create_target_angle(label_boxes, gt_masked)

            # mask the predicted outputs
            obj_masked, angles_masked, offsets_masked = obj_score[mb_mask], ang_scores[mb_mask], offset_scores[mb_mask]

            avod_cls_loss = self.classification_loss(obj_masked, gt_masked_cls)
            avod_loc_loss = self.localization_loss(offsets_masked, offsets_gt_masked, mask=cls_masked.bool())
            avod_angle_loss = self.orientation_loss(angles_masked, gt_angle_vec, mask=cls_masked.bool())

            final_loss = (avod_cls_loss + avod_angle_loss + avod_loc_loss + rpn_loss).float()

        return top_scores_soft, (top_pred_anchors, predictions_4c[top_indices], predictions_box[top_indices]), \
               top_pred_orientations, final_loss



@export
class AVOD(nn.Module):
    _path_drop_ = 'path_drop'

    def __init__(self, config=None):
        super(AVOD, self).__init__()

        if config is None:
            raise Exception('A config file must be provided')
        self.feature_extractors = FeatureExtractors(config, batch_norm=True)
        self.rpn = RPN(config)
        self.second_stage_detector = SecondStageDetector(config)
        self.drop_probabilities = config.get(self._path_drop_, [1., 1.])

    def create_path_drop_mask(self, device, disable_path_drop=True):
        if disable_path_drop or self.drop_probabilities[0] == self.drop_probabilities[1] == 1.:
            return torch.tensor([1.0], device=device), torch.tensor([1.0], device=device)
        values = np.random.uniform(size=3)

        img_mask = torch.tensor([1.0], device=device) if values[0] < self.drop_probabilities[0] else \
            torch.tensor([0.0], device=device)
        bev_mask = torch.tensor([1.0], device=device) if values[1] < self.drop_probabilities[1] else \
            torch.tensor([0.0], device=device)

        choice = torch.logical_or(img_mask, bev_mask).bool()

        img_mask_2 = torch.tensor([1.0], device=device) if values[2] > 0.5 else torch.tensor([0.0], device=device)
        bev_mask_2 = torch.tensor([1.0], device=device) if values[2] <= 0.5 else torch.tensor([0.0], device=device)

        img_mask = img_mask if choice & torch.as_tensor([True], device=device) else img_mask_2
        bev_mask = bev_mask if choice & torch.as_tensor([True], device=device) else bev_mask_2
        # if img_mask.item() == 0.0 and bev_mask.item() == 0.0:
        #     print()
        return img_mask, bev_mask

    def forward(self, x1, x2, anchors, mask, calibration_dict, ground_plane, targets, image_shape,
                enable_path_drop=False):
        # epoch = kwargs.get('val', True)
        targets_rpn = None
        targets_second_stage_detector = None
        if targets:
            targets_rpn, targets_second_stage_detector = targets[:2], targets[2:]

        img_map, bev_map = self.feature_extractors(x1, x2)

        img_mask, bev_mask = self.create_path_drop_mask(img_map.device, not enable_path_drop)

        top_anchors, top_obj_scores, rpn_loss = self.rpn(img_map, bev_map, anchors, mask, targets_rpn,
                                                         img_mask=img_mask, bev_mask=bev_mask)

        top_scores, top_boxes, top_orientations, final_loss = self.second_stage_detector(img_map, bev_map,
                                                                                         top_anchors,
                                                                                         image_shape,
                                                                                         calibration_dict,
                                                                                         ground_plane,
                                                                                         rpn_loss,
                                                                                         targets_second_stage_detector,
                                                                                         img_mask=img_mask,
                                                                                         bev_mask=bev_mask)
        # first tuple are the rpn predictions;second tuple contains the ssd's predictions
        return (top_anchors, top_obj_scores), (top_scores, top_boxes, top_orientations), final_loss


@export
class RPN2(nn.Module):
    _roi_type_dict = {'roi_align': RoIAlign, 'roi_pool': RoIPool}
    _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 2, 'spatial_scale': 1.},
                         'roi_pool': {'spatial_scale': 1.}}
    _batch_normalization_ = 'batch_normalization'

    _rpn_key_ = 'rpn'
    _input_channels_key_ = 'input_channels'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _pos_iou_range_ = 'iou_pos_range'
    _neg_iou_range_ = 'iou_neg_range'
    _nms_size_key_ = 'nms_size'
    _predictors_key_ = 'predictors'
    _mini_batch_size_key_ = 'mini_batch_size'
    _regression_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'

    def __init__(self, config=None):
        super(RPN2, self).__init__()

        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._rpn_key_)

        self.input_channels = config.get(self._input_channels_key_)
        self.pos_iou_range = config.get(self._pos_iou_range_)
        self.neg_iou_range = config.get(self._neg_iou_range_)
        self.mini_batch_size = config.get(self._mini_batch_size_key_)

        self.roi_size = config.get(self._roi_size_key_, 3)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.8)
        self.nms_size = config.get(self._nms_size_key_, 1024)
        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)
        self.roi_layer_bev = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)


        self.conv11_img = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=True,
                                    activation='relu')

        self.conv11_bev = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=True,
                                    activation='relu')

        self.fusion = Fusion()

        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.regression_weight = config.get(self._regression_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)

        self.regression_loss = WeightedSmoothL1Loss(self.regression_weight)
        self.classification_loss = SoftmaxCELoss(self.classification_weight)
        # self.classification_loss = nn.BCEWithLogitsLoss(torch.as_tensor(self.classification_weight))

    def forward(self, img_map, bev_map, anchors, gt_anchors_bev=None, targets=None, img_mask=None, bev_mask=None):
        # the anchors(axis aligned format) that were calculated by the dataset
        anchors_img, anchors_bev, filtered_anchors = anchors

        #  feed RPN with the previously acquired feature maps
        # RPN FOLLOWS
        # first there is a 1x1 conv layer followed by a ROI Pooling layer
        x1 = self.conv11_img(img_map)
        x2 = self.conv11_bev(bev_map)
        # crop and resize operation
        # ROI Pooling/Align https://pytorch.org/vision/stable/ops.html
        # extract feature regions using the projected anchors
        # extracted regions are of 3x3 size
        # anchors_img, anchors_bev represent the filtered anchors
        # projected into image and bev spaces respectively
        # img_mask, bev_mask are either .0 or 1. and are used for the path drop functionality i.e. when one of them
        # is .0 the equivalent path gets dropped;only one of them can be .0
        # roi_rgb = self.roi_layer_img(img_mask * x1, anchors_img, output_size=self.roi_size, **self.roi_function_params)
        # roi_bev = self.roi_layer_bev(bev_mask * x2, anchors_bev, output_size=self.roi_size, **self.roi_function_params)
        roi_rgb = self.roi_layer_img(img_mask * x1, anchors_img)
        roi_bev = self.roi_layer_bev(bev_mask * x2, anchors_bev)

        # Fusion
        # the feature crops of the previous step are fused using an element-wise mean operation
        fused = self.fusion(roi_rgb, roi_bev, img_mask + bev_mask)
        # Fully connected layers
        # instead of fully connected layers convolutional ones are  being used
        objectness, offsets = self.predictor(fused)
        # obtain the predicted anchors(using offsets and filtered anchors) and project them to bev space for nms
        reg_anchors = offset_predictions_to_anchors(offsets, filtered_anchors)

        bev_proposals = project_to_bev(reg_anchors, self.area_extents, normalize=False)

        # apply NMS to obtain the 1024 proposals with the higher score
        top_proposal_indices = nms(bev_proposals, F.softmax(objectness, dim=1)[:, 1],
                                   self.nms_threshold)[:self.nms_size]

        rpn_loss = None
        if targets is not None:
            gt_offsets, gt_objectness = targets

            mask, pos_mask = create_mini_batch_mask_rpn(gt_anchors_bev=gt_anchors_bev,
                                                        anchors_bev=anchors_bev[:, 1:],
                                                        pos_range=self.pos_iou_range,
                                                        neg_range=self.neg_iou_range,
                                                        size=self.mini_batch_size)

            rpn_loss = self.regression_loss(offsets[mask], gt_offsets[mask], objectness_gt=gt_objectness[mask]) + \
                       self.classification_loss(objectness[mask], gt_objectness[mask])

        return reg_anchors[top_proposal_indices], objectness[top_proposal_indices], rpn_loss


class SecondStageDetector2(nn.Module):
    _roi_type_dict = {'roi_align': RoIAlign, 'roi_pool': RoIPool}
    # _roi_funct_params = {'roi_align': {'aligned': True, 'sampling_ratio': 6}, 'roi_pool': {}}
    _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 6, 'spatial_scale': 1.},
                         'roi_pool': {'spatial_scale': 1.}}
    _batch_normalization_ = 'batch_normalization'

    _second_stage_detector_key_ = 'second_stage_detector'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_size_key_ = 'nms_size'
    _mini_batch_size_key_ = 'mini_batch_size'
    _predictors_key_ = 'predictors'
    _localization_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'
    _orientation_weight_key_ = 'orientation_weight'

    def __init__(self, config=None):
        super(SecondStageDetector2, self).__init__()

        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._second_stage_detector_key_)

        self.roi_size = config.get(self._roi_size_key_, 7)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.01)
        self.nms_size = config.get(self._nms_size_key_, 100)
        self.mini_batch_size = config.get(self._mini_batch_size_key_, 1024)

        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)
        self.roi_layer_bev = self._roi_type_dict[roi_type](self.roi_size, **self.roi_function_params)


        self.fusion = Fusion()
        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.localization_weight = config.get(self._localization_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)
        self.orientation_weight = config.get(self._orientation_weight_key_, 1.0)
        # losses
        self.classification_loss = SoftmaxCELoss(self.classification_weight)
        self.localization_loss = WeightedSmoothL1Loss(self.localization_weight)
        self.orientation_loss = WeightedSmoothL1Loss(self.orientation_weight)

    def forward(self, img_feat_map, bev_feat_map, top_anchors, image_shape, calibration_dict,
                ground_plane, rpn_loss=None, targets=None, img_mask=None, bev_mask=None, logger=None):
        bev_ins = project_to_bev(top_anchors, self.area_extents, add_zero_col=True)
        #
        rgb_ins = project_axis_aligned_to_img(top_anchors, image_shape, calibration_dict[0], normalize=False,
                                              add_zero_col=True)

        # ROI Pooling
        rois_rgb = self.roi_layer_img(img_mask * img_feat_map, rgb_ins)
        rois_bev = self.roi_layer_bev(bev_mask * bev_feat_map, bev_ins)
        # Fusion
        fused_avod = self.fusion(rois_rgb, rois_bev, img_mask + bev_mask)

        # Feed fully connected layers with feature crops
        obj_score, offset_scores, ang_scores = self.predictor(fused_avod)

        obj_score_softmax = F.softmax(obj_score, dim=1)

        orientations = convert_angle_to_orientation(ang_scores)

        predictions_4c, predictions_box, pred_anchors, proposals_4cp_format = proposals_to_4c_box_al(top_anchors,
                                                                                                     offset_scores,
                                                                                                     ground_plane)

        pred_bev = project_to_bev(pred_anchors, self.area_extents, normalize=False)
        top_pred_scores, _ = torch.max(obj_score[:, 1:], dim=1)
        top_indices = nms(pred_bev, top_pred_scores, self.nms_threshold)[:self.nms_size]

        # top_scores = obj_score[top_indices]
        top_scores_soft = obj_score_softmax[top_indices]
        top_pred_anchors = pred_anchors[top_indices]
        top_pred_orientations = orientations[top_indices]

        final_loss = None

        if rpn_loss:
            gt_anchors_bev, label_boxes, label_cls_ids = targets
            mb_mask, cls_masked, gt_masked = create_mini_batch_mask(gt_anchors_bev, bev_ins[:, 1:], label_cls_ids,
                                                                    size=self.mini_batch_size)
            # Create the target vectors

            # receives the masked proposals in 4cp format along with the labels and creates the target offsets
            # that will be used from the localization loss
            offsets_gt_masked = create_target_offsets(label_boxes[gt_masked], proposals_4cp_format[mb_mask],
                                                      ground_plane)

            gt_masked_cls = create_one_hot(cls_masked, neg_val=0.0010000000474974513)

            gt_angle_vec = create_target_angle(label_boxes, gt_masked)

            # mask the predicted outputs
            obj_masked, angles_masked, offsets_masked = obj_score[mb_mask], ang_scores[mb_mask], offset_scores[mb_mask]

            avod_cls_loss = self.classification_loss(obj_masked, gt_masked_cls)
            avod_loc_loss = self.localization_loss(offsets_masked, offsets_gt_masked, mask=cls_masked.bool())
            avod_angle_loss = self.orientation_loss(angles_masked, gt_angle_vec, mask=cls_masked.bool())

            final_loss = (avod_cls_loss + avod_angle_loss + avod_loc_loss + rpn_loss).float()

        return top_scores_soft, (top_pred_anchors, predictions_4c[top_indices], predictions_box[top_indices]), \
               top_pred_orientations, final_loss



@export
class AVOD2(nn.Module):
    _path_drop_ = 'path_drop'

    def __init__(self, config=None):
        super(AVOD2, self).__init__()

        if config is None:
            raise Exception('A config file must be provided')
        self.feature_extractors = FeatureExtractors(config, batch_norm=True)
        self.rpn = RPN2(config)
        self.second_stage_detector = SecondStageDetector2(config)
        self.drop_probabilities = config.get(self._path_drop_, [1., 1.])

    def create_path_drop_mask(self, device, disable_path_drop=True):
        if disable_path_drop or self.drop_probabilities[0] == self.drop_probabilities[1] == 1.:
            return torch.tensor([1.0], device=device), torch.tensor([1.0], device=device)
        values = np.random.uniform(size=3)

        img_mask = torch.tensor([1.0], device=device) if values[0] < self.drop_probabilities[0] else \
            torch.tensor([0.0], device=device)
        bev_mask = torch.tensor([1.0], device=device) if values[1] < self.drop_probabilities[1] else \
            torch.tensor([0.0], device=device)

        choice = torch.logical_or(img_mask, bev_mask).bool()

        img_mask_2 = torch.tensor([1.0], device=device) if values[2] > 0.5 else torch.tensor([0.0], device=device)
        bev_mask_2 = torch.tensor([1.0], device=device) if values[2] <= 0.5 else torch.tensor([0.0], device=device)

        img_mask = img_mask if choice & torch.as_tensor([True], device=device) else img_mask_2
        bev_mask = bev_mask if choice & torch.as_tensor([True], device=device) else bev_mask_2

        return img_mask, bev_mask

    def forward(self, x1, x2, anchors, calibration_dict, ground_plane, targets, image_shape,
                enable_path_drop=False):
        targets_rpn = None
        targets_second_stage_detector = None
        if targets:
            targets_rpn, targets_second_stage_detector = targets[:2], targets[2:]

        img_map, bev_map = self.feature_extractors(x1, x2)

        img_mask, bev_mask = self.create_path_drop_mask(img_map.device, not enable_path_drop)
        top_anchors, top_obj_scores, rpn_loss = self.rpn(img_map, bev_map, anchors, targets_second_stage_detector[0],
                                                         targets_rpn,
                                                         img_mask=img_mask, bev_mask=bev_mask)

        top_scores, top_boxes, top_orientations, final_loss = self.second_stage_detector(img_map, bev_map,
                                                                                         top_anchors,
                                                                                         image_shape,
                                                                                         calibration_dict,
                                                                                         ground_plane,
                                                                                         rpn_loss,
                                                                                         targets_second_stage_detector,
                                                                                         img_mask=img_mask,
                                                                                         bev_mask=bev_mask)
        # first tuple are the rpn predictions;second tuple contains the ssd's predictions
        return (top_anchors, top_obj_scores), (top_scores, top_boxes, top_orientations), final_loss
