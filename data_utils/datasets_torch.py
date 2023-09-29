import torch
import numpy as np

from general_utils import export
from data_utils.dataset_base import KITTIDataset
from data_utils.transforms import BaseKittiImageTransform, KittiPCTransformTorch, Compose

__all__ = []


@export
class KITTIDatasetTorch(KITTIDataset):

    def __init__(self, mode='train', config=None):
        super(KITTIDatasetTorch, self).__init__(mode=mode, config=config)
        self.preprocess_point_cloud = KittiPCTransformTorch(self.transformation_config)

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


        # resizing moved to transforms
        images = torch.stack(imgs)

        bev_maps = torch.stack(bevs)

        filtered_anchors = torch.cat([torch.as_tensor(anchor) for anchor in filtered_anchors]).float()

        offsets = torch.cat([torch.as_tensor(offset) for offset in offsets]).float()
        # adding batch index to anchors and converting to tensor
        bev_anchors = torch.cat(
            [torch.as_tensor(np.append(np.zeros((len(bev_anchor), 1), dtype=np.float32) + i, bev_anchor, axis=1))
             for i, bev_anchor in enumerate(bev_anchors)]).float()
        rgb_anchors = torch.cat(
            [torch.as_tensor(np.append(np.zeros((len(rgb_anchor), 1), dtype=np.float32) + i, rgb_anchor, axis=1))
             for i, rgb_anchor in enumerate(rgb_anchors)]).float()

        obj_tensor = torch.cat([torch.as_tensor(obj_score) for obj_score in objectness])

        mask_tensor = torch.cat([torch.as_tensor(i) for i in mask])

        anchors_gt_bev = torch.cat([torch.as_tensor(anchor) for anchor in anchors_gt_bev])

        label_cls_ids = torch.cat([torch.as_tensor(cls_id.astype(np.int8)) for cls_id in label_cls_ids])

        labels = torch.cat([torch.as_tensor(label) for label in labels])

        ground_plane = torch.cat([torch.as_tensor(gp) for gp in ground_plane])

        image_shape = torch.stack([torch.as_tensor(np.array(shape).reshape(-1, 3)) for shape in current_image_shape])

        return images, bev_maps, (rgb_anchors, bev_anchors, filtered_anchors), mask_tensor, calib_dict, \
               ground_plane, (offsets, obj_tensor, anchors_gt_bev, labels, label_cls_ids), image_shape, names

