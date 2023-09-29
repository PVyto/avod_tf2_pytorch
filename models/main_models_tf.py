import numpy as np
import tensorflow as tf
from general_utils import export
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from .helper_functions import generate_multiple_input_dims
from .building_blocks_tf import FeatureExtractor, ConvBlock, Fusion, Predictor
from utils.box_utils_tf import offset_predictions_to_anchors, project_to_bev, project_axis_aligned_to_img, \
    convert_angle_to_orientation, proposals_to_4c_box_al
from data_utils.dataset_utils_tf import create_mini_batch_mask, create_target_offsets, create_target_angle, \
    create_one_hot
from utils.losses_tf import WeightedSmoothL1Loss, SoftmaxCELoss

nn = tf.keras.layers


class FeatureExtractors(Layer):
    _feature_extractors_config_ = 'feature_extractors'
    _img_input_channels_ = 'img_input_channels'
    _bev_input_channels_ = 'bev_input_channels'
    _output_channels_ = 'out_channels'
    _activation_function_ = 'activation_function'
    _batch_normalization_ = 'batch_normalization'
    _weight_initialization_method_ = 'weight_initialization_method'

    def __init__(self, config=None, batch_norm=False):
        # maybe change the signature since batch_norm is enabled via the configuration file
        super(FeatureExtractors, self).__init__()
        config = config[self._feature_extractors_config_]

        image_input_channels = config[self._img_input_channels_]
        bev_input_channels = config[self._bev_input_channels_]
        output_channels = config[self._output_channels_]
        activation = config[self._activation_function_]
        enable_batch_norm = config[self._batch_normalization_]
        weight_init_method = config.get(self._weight_initialization_method_, 'glorot_uniform')
        input_dims_img, input_dims_bev = generate_multiple_input_dims([[360, 1200], [704, 800]])

        self.feature_extractor_img = FeatureExtractor(image_input_channels, output_channels,
                                                      batch_norm=enable_batch_norm,
                                                      activation=activation, input_dims=input_dims_img,
                                                      weight_init_method=weight_init_method)
        self.feature_extractor_bev = FeatureExtractor(bev_input_channels, output_channels, batch_norm=enable_batch_norm,
                                                      activation=activation, input_dims=input_dims_bev,
                                                      weight_init_method=weight_init_method)

    def call(self, x1, x2):
        return self.feature_extractor_img(x1), self.feature_extractor_bev(x2)[:, 4:, :, :]


class RPN(Layer):
    _roi_type_dict = {'roi_align': tf.image.crop_and_resize, 'roi_pool': tf.image.crop_and_resize}

    _roi_layer_params = {'roi_align': {},
                         'roi_pool': {}}
    _batch_normalization_ = 'batch_normalization'

    _rpn_key_ = 'rpn'
    _input_channels_key_ = 'input_channels'
    _roi_size_key_ = 'roi_size'
    _roi_type_key_ = 'roi_layer'
    _nms_threshold_key_ = 'nms_threshold'
    _nms_train_size_key_ = 'nms_size_train'
    _nms_test_size_key_ = 'nms_size_test'
    _predictors_key_ = 'predictors'
    _mini_batch_size_key_ = 'mini_batch_size'
    _regression_weight_key_ = 'regression_weight'
    _classification_weight_key_ = 'classification_weight'

    def __init__(self, config=None, input_dims=None):
        super(RPN, self).__init__()

        if input_dims is None:
            raise Exception('input_dims must not be None')

        input_dims_img, input_dims_bev = input_dims
        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._rpn_key_)

        self.input_channels = config.get(self._input_channels_key_)
        self.roi_size = config.get(self._roi_size_key_, 3)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.8)
        self.nms_size_train = config.get(self._nms_train_size_key_, 1024)
        self.nms_size_test = config.get(self._nms_test_size_key_, 300)  # 300 for cars 1024 for pedestrians and cyclists
        self.nms_size = self.nms_size_train
        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type]
        self.roi_layer_bev = self._roi_type_dict[roi_type]
        enable_batch_norm = config.get(self._batch_normalization_)



        self.conv11_img = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=enable_batch_norm,
                                    activation='relu', dim=input_dims_img)

        self.conv11_bev = ConvBlock(self.input_channels, out_channels=1, kernel_size=1, batch_norm=enable_batch_norm,
                                    activation='relu', dim=input_dims_bev)

        self.fusion = Fusion()

        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.regression_weight = config.get(self._regression_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)

        self.regression_loss = WeightedSmoothL1Loss(self.regression_weight)
        self.classification_loss = SoftmaxCELoss(self.classification_weight)

    def call(self, img_map, bev_map, anchors, mask=None, targets=None, img_mask=None, bev_mask=None):
        # the anchors(axis aligned format) that were calculated by the dataset
        anchors_img, anchors_bev, filtered_anchors = anchors
        anchors_img_crop_resize = tf.gather(anchors_img, [2, 1, 4, 3], axis=1)
        anchors_bev_crop_resize = tf.gather(anchors_bev, [2, 1, 4, 3], axis=1)

        #  feed RPN with the previously acquired feature maps
        # RPN FOLLOWS
        # first there is a 1x1 conv layer followed by a ROI Pooling layer
        x1 = self.conv11_img(img_map)
        x2 = self.conv11_bev(bev_map)
        # crop and resize operation
        # extract feature regions using the projected anchors
        # extracted regions are of 3x3 size
        # anchors_img, anchors_bev represent the filtered anchors
        # projected into image and bev spaces respectively
        # img_mask, bev_mask are either .0 or 1. and are used for the path drop functionality i.e. when one of them
        # is .0 the equivalent path gets dropped;only one of them can be .0

        roi_rgb = self.roi_layer_img(img_mask * x1, boxes=anchors_img_crop_resize,
                                     box_indices=tf.cast(anchors_img[:, 0], dtype=tf.int32),
                                     crop_size=2 * [self.roi_size])
        roi_bev = self.roi_layer_bev(bev_mask * x2, boxes=anchors_bev_crop_resize,
                                     box_indices=tf.cast(anchors_bev[:, 0], dtype=tf.int32),
                                     crop_size=2 * [self.roi_size])

        # Fusion
        # the feature crops of the previous step are fused using an element-wise mean operation
        fused = self.fusion(roi_rgb, roi_bev, img_mask + bev_mask)
        # Fully connected layers
        # instead of fully connected layers convolutional ones can be used
        objectness, offsets = self.predictor(fused)
        # return objectness, offsets
        # obtain the predicted anchors(using offsets and filtered anchors) and project them to bev space for nms
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        reg_anchors = offset_predictions_to_anchors(offsets, filtered_anchors)
        #
        bev_proposals = project_to_bev(reg_anchors, self.area_extents, normalize=False)
        #
        # # apply NMS to obtain the 1024 proposals with the higher score
        top_proposal_indices = tf.image.non_max_suppression(bev_proposals, tf.nn.softmax(objectness, axis=1)[:, 1],
                                                            iou_threshold=self.nms_threshold,
                                                            max_output_size=self.nms_size)
        rpn_loss = None
        if targets is not None:
            gt_offsets, gt_objectness = targets
            offsets_masked, gt_offsets_masked = tf.boolean_mask(offsets, mask), tf.boolean_mask(gt_offsets, mask)
            objectness_masked, gt_objectness_masked = tf.boolean_mask(objectness, mask), \
                                                      tf.boolean_mask(gt_objectness, mask)

            reg_loss = self.regression_loss(offsets_masked, gt_offsets_masked, objectness_gt=gt_objectness_masked)
            cls_loss = self.classification_loss(objectness_masked, gt_objectness_masked)
            rpn_loss = reg_loss + cls_loss

        return tf.gather(reg_anchors, top_proposal_indices, axis=0), \
               tf.gather(objectness, top_proposal_indices, axis=0), \
               rpn_loss

    def eval(self):
        self.nms_size = self.nms_size_test

    def train(self):
        self.nms_size = self.nms_size_train


class SecondStageDetector(Layer):
    _roi_type_dict = {'roi_align': tf.image.crop_and_resize, 'roi_pool': tf.image.crop_and_resize}
    _roi_layer_params = {'roi_align': {'aligned': True, 'sampling_ratio': 6, 'spatial_scale': 1.},
                         'roi_pool': {'spatial_scale': 1.}}
    _batch_normalization_ = 'batch_normalization'
    _pos_iou_range_ = 'iou_pos_range'
    _neg_iou_range_ = 'iou_neg_range'
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
    _number_of_classes_ = 'num_classes'

    def __init__(self, config=None):
        super(SecondStageDetector, self).__init__()

        self.area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(3, 2)

        config = config.get(self._second_stage_detector_key_)

        self.num_classes = config.get(self._number_of_classes_)
        self.pos_iou_range = config.get(self._pos_iou_range_)
        self.neg_iou_range = config.get(self._neg_iou_range_)

        self.roi_size = config.get(self._roi_size_key_, 7)
        self.nms_threshold = config.get(self._nms_threshold_key_, 0.01)
        self.nms_size = config.get(self._nms_size_key_, 100)
        self.mini_batch_size = config.get(self._mini_batch_size_key_, 1024)

        self.roi_function_params = self._roi_layer_params[config.get(self._roi_type_key_, 'roi_align')]
        roi_type = config.get(self._roi_type_key_, 'roi_align')
        self.roi_layer_img = self._roi_type_dict[roi_type]  # (self.roi_size, **self.roi_function_params)
        self.roi_layer_bev = self._roi_type_dict[roi_type]  # (self.roi_size, **self.roi_function_params)


        self.fusion = Fusion()
        self.predictor = Predictor(config.get(self._predictors_key_), self.roi_size)

        self.localization_weight = config.get(self._localization_weight_key_, 1.0)
        self.classification_weight = config.get(self._classification_weight_key_, 1.0)
        self.orientation_weight = config.get(self._orientation_weight_key_, 1.0)
        # losses
        self.classification_loss = SoftmaxCELoss(self.classification_weight)
        self.localization_loss = WeightedSmoothL1Loss(self.localization_weight)
        self.orientation_loss = WeightedSmoothL1Loss(self.orientation_weight)

    def call(self, img_feat_map, bev_feat_map, top_anchors, image_shape, calibration_dict,
             ground_plane, rpn_loss=None, targets=None, img_mask=None, bev_mask=None, calculate_loss=False):
        bev_ins = project_to_bev(top_anchors, self.area_extents, add_zero_col=False, normalize=True)
        rgb_ins = project_axis_aligned_to_img(top_anchors, image_shape, calibration_dict[0], normalize=True,
                                              add_zero_col=False)
        anchors_img_crop_resize = tf.gather(rgb_ins, [1, 0, 3, 2], axis=1)
        anchors_bev_crop_resize = tf.gather(bev_ins, [1, 0, 3, 2], axis=1)

        # ROI Pooling
        rois_rgb = self.roi_layer_img(img_mask * img_feat_map, boxes=anchors_img_crop_resize,
                                      box_indices=tf.zeros(len(rgb_ins), dtype=tf.int32),
                                      crop_size=2 * [self.roi_size])
        rois_bev = self.roi_layer_bev(bev_mask * bev_feat_map, boxes=anchors_bev_crop_resize,
                                      box_indices=tf.zeros(len(bev_ins), dtype=tf.int32),
                                      crop_size=2 * [self.roi_size])
        # Fusion
        fused_avod = self.fusion(rois_rgb, rois_bev, img_mask + bev_mask)

        # Feed fully connected layers with feature crops
        obj_score, offset_scores, ang_scores = self.predictor(fused_avod)

        obj_score_softmax = tf.nn.softmax(obj_score, axis=1)

        orientations = convert_angle_to_orientation(ang_scores)

        #
        predictions_4c, predictions_box, pred_anchors, proposals_4cp_format = proposals_to_4c_box_al(top_anchors,
                                                                                                     offset_scores,
                                                                                                     ground_plane)

        pred_bev = project_to_bev(pred_anchors, self.area_extents, normalize=False)
        top_pred_scores = tf.reduce_max(obj_score[:, 1:], axis=1)
        top_indices = tf.image.non_max_suppression(pred_bev, top_pred_scores, iou_threshold=self.nms_threshold,
                                                   max_output_size=self.nms_size)

        top_scores_soft = tf.gather(obj_score_softmax, top_indices, axis=0)
        top_pred_anchors = tf.gather(pred_anchors, top_indices, axis=0)
        top_pred_orientations = tf.gather(orientations, top_indices, axis=0)

        #
        final_loss = None
        # # if we are training the section below will be executed
        if rpn_loss and calculate_loss:
            bev_ins2 = project_to_bev(top_anchors, self.area_extents, add_zero_col=False, normalize=False)
            gt_anchors_bev, label_boxes, label_cls_ids = targets
            mb_mask, cls_masked, gt_masked = create_mini_batch_mask(gt_anchors_bev, bev_ins2, label_cls_ids,
                                                                    size=self.mini_batch_size,
                                                                    pos_range=self.pos_iou_range,
                                                                    neg_range=self.neg_iou_range)

            # Create the target vectors

            # receives the masked proposals in 4cp format along with the labels and creates the target offsets
            # that will be used for computing the localization loss
            offsets_gt_masked = create_target_offsets(tf.gather(label_boxes, gt_masked),
                                                      tf.boolean_mask(proposals_4cp_format, mb_mask),
                                                      ground_plane)
            #
            gt_masked_cls = create_one_hot(cls_masked, neg_val=0.0010000000474974513 / (self.num_classes - 1),
                                           num_classes=self.num_classes)
            #
            gt_angle_vec = create_target_angle(label_boxes, gt_masked)
            #
            #     # mask the predicted outputs
            obj_masked, angles_masked, offsets_masked = tf.boolean_mask(obj_score, mb_mask), \
                                                        tf.boolean_mask(ang_scores, mb_mask), \
                                                        tf.boolean_mask(offset_scores, mb_mask)
            #
            avod_cls_loss = self.classification_loss(obj_masked, gt_masked_cls)
            cls_masked_bool = tf.cast(cls_masked, tf.bool)
            avod_loc_loss = self.localization_loss(offsets_masked, offsets_gt_masked, mask=cls_masked_bool)
            avod_angle_loss = self.orientation_loss(angles_masked, gt_angle_vec, mask=cls_masked_bool)
            #
            final_loss = avod_cls_loss + avod_angle_loss + avod_loc_loss + rpn_loss
        #
        return top_scores_soft, \
               (top_pred_anchors, tf.gather(predictions_4c, top_indices, axis=0),
                tf.gather(predictions_box, top_indices, axis=0)), top_pred_orientations, final_loss



@export
class Avod(Model):
    _path_drop_ = 'path_drop'

    def __init__(self, config=None, rpn_input_dims=None):
        super(Avod, self).__init__()

        if rpn_input_dims is None:
            rpn_input_dims = []
        if config is None:
            raise Exception('A config file must be provided')
        self.feature_extractors = FeatureExtractors(config, batch_norm=True)
        self.rpn = RPN(config, input_dims=rpn_input_dims)
        self.second_stage_detector = SecondStageDetector(config)
        self.drop_probabilities = config.get(self._path_drop_, [1., 1.])
        self.calculate_loss = True

    def create_path_drop_mask(self, disable_path_drop=True):
        if disable_path_drop or self.drop_probabilities[0] == self.drop_probabilities[1] == 1.:
            return tf.constant(1.0), tf.constant(1.0)
        values = np.random.uniform(size=3)

        img_mask = tf.constant(1.0) if values[0] < self.drop_probabilities[0] else tf.constant(0.0)
        bev_mask = tf.constant(1.0) if values[1] < self.drop_probabilities[1] else tf.constant(0.0)
        img_mask = tf.cast(img_mask, tf.bool)
        bev_mask = tf.cast(bev_mask, tf.bool)
        choice = tf.logical_or(img_mask, bev_mask)

        img_mask_2 = tf.constant(1.0) if values[2] > 0.5 else tf.constant(0.0)
        bev_mask_2 = tf.constant(1.0) if values[2] <= 0.5 else tf.constant(0.0)

        img_mask = img_mask if choice & tf.constant(True) else img_mask_2
        bev_mask = bev_mask if choice & tf.constant(True) else bev_mask_2
        # if img_mask.item() == 0.0 and bev_mask.item() == 0.0:
        #     print()
        return tf.cast(img_mask, tf.float32), tf.cast(bev_mask, tf.float32)

    def call(self, x1, x2, anchors, mask, calibration_dict, ground_plane, targets, image_shape, enable_path_drop=False):

        targets_rpn = None
        targets_second_stage_detector = None
        if targets:
            targets_rpn, targets_second_stage_detector = targets[:2], targets[2:]

        img_map, bev_map = self.feature_extractors(x1, x2)

        img_mask, bev_mask = self.create_path_drop_mask(not enable_path_drop)

        top_anchors, top_obj_scores, rpn_loss = self.rpn(img_map, bev_map, anchors, mask, targets_rpn,
                                                         img_mask=img_mask, bev_mask=bev_mask)
        # print('top_anchors: ', top_anchors)
        top_scores, top_boxes, top_orientations, final_loss = self.second_stage_detector(img_map, bev_map,
                                                                                         top_anchors,
                                                                                         image_shape,
                                                                                         calibration_dict,
                                                                                         ground_plane,
                                                                                         rpn_loss,
                                                                                         targets_second_stage_detector,
                                                                                         img_mask=img_mask,
                                                                                         bev_mask=bev_mask,
                                                                                         calculate_loss=self.calculate_loss)
        # first tuple are the rpn predictions;second tuple contains the ssd's predictions
        return (top_anchors, top_obj_scores), (top_scores, top_boxes, top_orientations), final_loss

    def eval(self):
        self.rpn.eval()
        self.calculate_loss = False

    def train(self):
        self.rpn.train()
        self.calculate_loss = True
