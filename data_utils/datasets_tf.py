import numpy as np
import tensorflow as tf

from general_utils import export
from data_utils.dataset_base import KITTIDataset
from data_utils.transforms import BaseKittiImageTransformTf, KittiPCTransformTf, Compose

__all__ = []

KittiTransform = KittiPCTransformTf
BaseKittiImageTransform = BaseKittiImageTransformTf


@export
class KITTIDatasetTf(KITTIDataset):

    def __init__(self, mode='train', config=None):
        super(KITTIDatasetTf, self).__init__(mode=mode, config=config)

        self.preprocess_point_cloud = KittiTransform(self.transformation_config)

    def _prepare_transforms(self):
        self._transforms = [self._available_transforms[t]() for t in self.image_transformation_config]
        if self.mode != 'train':
            self._transform_image = BaseKittiImageTransform()
        else:
            self._transform_image = Compose(self._transforms)

    @staticmethod
    def collate_fn(batch):

        imgs, bevs, anchors_gt_bev, filtered_anchors, bev_anchors, rgb_anchors, ious, offsets, objectness, mask, \
        calib_dict, label_cls_ids, labels, ground_plane, current_image_shape, names = list(zip(*batch))
        images = tf.stack(imgs, axis=0)

        bev_maps = tf.stack(bevs, axis=0)

        filtered_anchors = tf.concat([tf.convert_to_tensor(anchor) for anchor in filtered_anchors], axis=0)
        filtered_anchors = tf.cast(filtered_anchors, tf.float32)
        offsets = tf.cast(tf.concat([tf.convert_to_tensor(offset) for offset in offsets], axis=0), tf.float32)
        # adding batch index to anchors and converting to tensor
        bev_anchors = tf.concat(
            [tf.convert_to_tensor(np.append(np.zeros((len(bev_anchor), 1), dtype=np.float32) + i, bev_anchor, axis=1))
             for i, bev_anchor in enumerate(bev_anchors)], axis=0)

        bev_anchors = tf.cast(bev_anchors, dtype=tf.float32)
        rgb_anchors = tf.concat(
            [tf.convert_to_tensor(np.append(np.zeros((len(rgb_anchor), 1), dtype=np.float32) + i, rgb_anchor, axis=1))
             for i, rgb_anchor in enumerate(rgb_anchors)], axis=0)
        rgb_anchors = tf.cast(rgb_anchors, dtype=tf.float32)

        obj_tensor = tf.concat([tf.convert_to_tensor(obj_score) for obj_score in objectness], axis=0)

        mask_tensor = tf.concat([tf.convert_to_tensor(i) for i in mask], axis=0)

        anchors_gt_bev = tf.concat([tf.convert_to_tensor(anchor) for anchor in anchors_gt_bev], axis=0)

        label_cls_ids = tf.concat([tf.convert_to_tensor(cls_id.astype(np.int8)) for cls_id in label_cls_ids], axis=0)

        labels = tf.concat([tf.convert_to_tensor(label) for label in labels], axis=0)

        ground_plane = tf.concat([tf.convert_to_tensor(gp) for gp in ground_plane], axis=0)

        image_shape = tf.stack([tf.convert_to_tensor(np.array(shape).reshape(-1, 3)) for shape in current_image_shape])

        return images, bev_maps, (rgb_anchors, bev_anchors, filtered_anchors), mask_tensor, calib_dict, \
               ground_plane, (offsets, obj_tensor, anchors_gt_bev, labels, label_cls_ids), image_shape, names


class CustomDatasetTf(KITTIDatasetTf):
    def __init__(self, mode='train', config=None):
        super(CustomDatasetTf, self).__init__(mode=mode, config=config)

    def __getitem__(self, idx):
        image, bev_maps, anchors_gt_bev, filtered_anchors, bev_anchors, rgb_anchors, ious, offsets, objectness, mask, \
        calib_dict, label_cls_ids, labels, ground_plane, self.current_image_shape, self.sample_name = \
            super().__getitem__(idx)
        return image, bev_maps, anchors_gt_bev, filtered_anchors, bev_anchors, rgb_anchors, ious, offsets, objectness, mask, \
               calib_dict['frame_calib_matrix'][
                   2], label_cls_ids, labels, ground_plane, self.current_image_shape, self.sample_name

    @staticmethod
    def collate_tf(images, bev_maps, anchors_gt_bev, filtered_anchors, bev_anchors, rgb_anchors, ious, offsets,
                   objectness, mask, rect_mat, label_cls_ids, labels, ground_plane, image_shape, sample_name):
        images = tf.expand_dims(images, axis=0)
        bev_maps = tf.expand_dims(bev_maps, axis=0)
        #         anchors_gt_bev = tf.convert_to_tensor(anchors_gt_bev)
        img_anchors = tf.pad(rgb_anchors, [[0, 0], [1, 0]])
        bev_anchors = tf.pad(bev_anchors, [[0, 0], [1, 0]])
        anchors_gt_bev = tf.cast(anchors_gt_bev, tf.float32)
        image_shape = tf.expand_dims(tf.expand_dims(image_shape, axis=0), axis=0)
        anchors = (img_anchors, bev_anchors, filtered_anchors)
        targets = (offsets, objectness, anchors_gt_bev, labels, label_cls_ids)
        return images, bev_maps, anchors, mask, rect_mat, ground_plane, targets, image_shape, sample_name
