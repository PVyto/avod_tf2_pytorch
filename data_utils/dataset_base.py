import os
import glob
import numpy as np
import pandas as pd

from skimage import io
from torch.utils.data import Dataset
from torch import load as torch_load
from configs.config import dataset_config
from utils.anchor_manipulator import AnchorManipulator
from general_utils import get_available_options_from_module

from utils.box_utils import calculate_iou, convert_box_to_axis_al_format, project_to_bev, project_axis_aligned_to_img
from data_utils.dataset_utils import load_annotations_par, create_one_hot, create_regression_offsets, exists, \
    get_iou_mask

from pathlib import Path


class KITTIDataset(Dataset):
    _experiment_ = 'experiment'

    _mode_ = 'mode'
    _base_dir_ = 'path'
    _image_dir_ = 'image_path'
    _label_dir_ = 'label_path'
    _planes_dir_ = 'plane_path'
    _annotation_prefix_ = 'annotation'
    _folder_mapping_ = 'folder_mapping'
    _point_cloud_dir_ = 'point_cloud_path'
    _calibration_dir_ = 'calibration_path'
    _neg_iou_range_ = 'iou_neg_range'
    _pos_iou_range_ = 'iou_pos_range'
    _classes_ = 'classes'
    _image_size_ = 'image_size'
    _difficulty_ = 'difficulty'
    _anchor_sizes_ = 'anchor_sizes'
    _num_clusters_ = 'num_anchors'
    _area_extents_ = 'area_extents'
    _anchor_stride_ = 'anchor_stride'
    _mini_batch_size_key_ = 'mini_batch_size'

    _transformation_config_ = 'transformation'
    _image_transform_config_ = 'transform'

    _available_transforms = get_available_options_from_module('data_utils.transforms')

    def __init__(self, mode='train', config=None):
        if mode not in {'train', 'test', 'val'}:
            raise Exception(
                'Mode must take one of the following values: train, test, val but the value "' + mode + '" was given')
        if config is None:
            config = dataset_config

        self.model = 'AVOD2'
        self.config = config
        self.name = config.get(self._experiment_, '')
        self.root_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        self.annotation_csv_dir = self.root_dir / 'annotations'

        self.mode = mode
        self.base_path = Path(os.path.expanduser(config[self._base_dir_]))
        # extract the sample names retaining the 6 digit format
        self.sample_list = ['{:0>6}'.format(i) for i in
                            pd.read_csv(self.base_path / (config[self._annotation_prefix_ + '_' + mode] + '.txt'),
                                        header=None).iloc[:, 0].to_list()]
        self.sample_names = []

        self.dataset = self.base_path / config[self._folder_mapping_][self.mode]
        self.img_dir = self.dataset / config[self._image_dir_]
        self.velo_dir = self.dataset / config[self._point_cloud_dir_]
        self.annotation_dir = self.dataset / config[self._label_dir_]
        self.calib_dir = self.dataset / config[self._calibration_dir_]
        self.planes_dir = self.dataset / config[self._planes_dir_]

        self.sample_name = ''
        self.current_calib = dict()
        self.current_ground_plane = []

        # images will be resized to this
        self.img_size = config[self._image_size_]
        self.current_image_shape = []
        self.classes = config.get(self._classes_)
        self.class_indices = [0] + list(range(1, len(self.classes) + 1))

        self.cls_dict = {'Bckg': 0}
        self.cls_dict.update({cls: i + 1 for i, cls in enumerate(self.classes)})
        self.inv_cls_dict = {v: k for k, v in self.cls_dict.items()}

        self.difficulty = config[self._difficulty_]
        self.anchor_stride = config[self._anchor_stride_]

        self.num_clusters = config[self._num_clusters_]
        self.anchor_size = np.array(config[self._anchor_sizes_]).reshape(2, -1)

        self.area_extents = np.array(config[self._transformation_config_].get(self._area_extents_)).reshape(3, 2)
        self.transformation_config = config[self._transformation_config_]
        self.image_transformation_config = config[self._image_transform_config_]

        self.rpn_neg_iou_range = config[self._neg_iou_range_]  # [0.0, 0.3]
        self.rpn_pos_iou_range = config[self._pos_iou_range_]  # [0.5, 1.0]
        self.mini_batch_size = config[self._mini_batch_size_key_]

        self._create_caching_folders()
        self._get_file_lists()
        self._load_annotations()
        self._prepare_transforms()

        # transforms point cloud into height and density maps

        self.anchor_generator = AnchorManipulator(stride=self.anchor_stride,
                                                  area_extents=self.area_extents)

    def _get_file_lists(self):
        self.imgs = sorted(self.img_dir.glob('*'))
        self.velo_files = sorted(self.velo_dir.glob('*'))
        self.annotation_files = sorted(self.annotation_dir.glob('*.txt'))
        self.calib_files = sorted(self.calib_dir.glob('*.txt'))
        self.plane_files = sorted(self.planes_dir.glob('*.txt'))
        if self.sample_list:
            self.sample_list = sorted(list(set(self.sample_list)))
            self.imgs = [self.img_dir / (i + '.png') for i in self.sample_list]
            self.velo_files = [self.velo_dir / (i + '.bin') for i in self.sample_list]
            self.annotation_files = [self.annotation_dir / (i + '.txt') for i in self.sample_list]
            self.calib_files = [self.calib_dir / (i + '.txt') for i in self.sample_list]
            self.plane_files = [self.planes_dir / (i + '.txt') for i in self.sample_list]
        self.sample_names = sorted([i.name.split('.')[0] for i in self.imgs])

    def _load_annotations(self):
        try:
            self.annotations = pd.read_csv(self.annotation_cache_folder / (self.name + '_' + self.mode + '.csv'))
        except FileNotFoundError:
            self.annotations = load_annotations_par(self.annotation_files)
            self.annotations.to_csv(self.annotation_cache_folder / (self.name + '_' + self.mode + '.csv'), index=False)
        if self.classes:
            # filter by class
            self.annotations = self.annotations[self.annotations['class'].isin(self.classes)]
            self._update_annotations()

    def _update_annotations(self):
        keys_ = [i.split('.')[0] for i in sorted(list(self.annotations['image_name'].unique()))]
        self.sample_names = keys_
        self.imgs = [img.expanduser() for img in self.imgs for _key in keys_ if _key in img.name]
        self.velo_files = [velo.expanduser() for velo in self.velo_files for _key in keys_ if _key in velo.name]
        self.calib_files = [calib.expanduser() for calib in self.calib_files for _key in keys_ if _key in calib.name]
        self.plane_files = [plane.expanduser() for plane in self.plane_files for _key in keys_ if _key in plane.name]
        for k, v in {k.name: i for i, k in enumerate(self.imgs)}.items():
            self.annotations.loc[self.annotations['image_name'] == k, 'idx'] = v

    def _create_caching_folders(self):
        self.caching_folder = self.dataset / 'cache'
        self.bev_maps_folder = self.root_dir / 'bev_maps'
        self.annotation_cache_folder = self.annotation_csv_dir / self.name
        os.makedirs(self.caching_folder, exist_ok=True)
        os.makedirs(self.annotation_cache_folder, exist_ok=True)
        # os.makedirs(self.bev_maps_folder, exist_ok=True)

    def _generate_anchors(self, ground_plane):
        return self.anchor_generator.generate_anchors_3d(ground_plane, self.anchor_size)

    def _filter_anchors(self, anchors, point_cloud, ground_plane):
        anchor_filter = self.anchor_generator.get_non_empty_anchors_filter(anchors, point_cloud, ground_plane,
                                                                           self.transformation_config['voxel_size'])
        return anchors[anchor_filter], anchor_filter

    def _anchors_to_2d(self, anchors, rotate=False):
        # converting to axis aligned format
        return convert_box_to_axis_al_format(anchors, rotate)

    def _extract_info_about_anchors(self, anchors, labels, labels_cls):
        axis_aligned_labels = convert_box_to_axis_al_format(labels, True)
        bev_anchors = project_to_bev(anchors, self.area_extents)
        bev_labels = project_to_bev(axis_aligned_labels, self.area_extents)
        iou_scores = np.zeros((len(bev_anchors)))
        offsets = np.zeros((len(bev_anchors), 6))
        class_idxs = np.zeros((len(bev_anchors)))
        for i, label in enumerate(bev_labels):
            class_idx = self.classes.index(labels_cls[i]) + 1
            ious = calculate_iou(label.reshape(1, -1), bev_anchors).squeeze()
            mask = ious > iou_scores
            iou_scores[mask] = ious[mask]
            anch_offsets = create_regression_offsets(anchors[mask], axis_aligned_labels[i].reshape(1, -1))
            offsets[mask] = anch_offsets
            class_idxs[mask] = class_idx
        return iou_scores, offsets, class_idxs

    def _filter_annotation_(self):
        diff_dict = {'0': [40, 0, 0.15], '1': [25, 1, 0.3], '2': [25, 2, 0.5]}
        if isinstance(self.difficulty, list):
            # 		check for validity
            self.difficulty = set(self.difficulty) & {0, 1, 2}
            self.difficulty = str(max(self.difficulty))
            self.annotations = self.annotations[
                self.annotations['occlusion'] <= diff_dict[self.difficulty][1] & self.annotations['truncation'] <=
                diff_dict[self.difficulty][2] & self.annotations['height']]

    def loader(self, load_funct):
        pass

    def _load_image(self, img_idx):
        image_path = self.imgs[img_idx]
        # print('Loading image named ', self.imgs[img_idx])

        image = io.imread(image_path)

        self.sample_name = image_path.name.split('.')[0]
        self.current_image_shape = image.shape

        return image.astype(np.float32)

    def _load_calibration(self, img_idx):
        calib_file_path = self.calib_files[img_idx]
        if exists(calib_file_path):
            calib_info_df = pd.read_csv(calib_file_path, header=None, sep=': | ', index_col=0, engine='python').T.loc[:,
                            ['P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam']]

            frame_calib_matrix = calib_info_df.loc[:, ['P0', 'P1', 'P2', 'P3']].to_numpy().T.reshape(4, 3, 4)

            rectification_matrix = calib_info_df.loc[:, ['R0_rect']].dropna().to_numpy().reshape(3, -1)

            velodyne_to_cam_matrix = calib_info_df.loc[:, ['Tr_velo_to_cam']].to_numpy().reshape(3, -1)

            self.current_calib = {'frame_calib_matrix': frame_calib_matrix.astype(np.float32),
                                  'rectification_matrix': rectification_matrix.astype(np.float32),
                                  'velodyne_to_cam_matrix': velodyne_to_cam_matrix.astype(np.float32)}

            return {'frame_calib_matrix': frame_calib_matrix.astype(np.float32),
                    'rectification_matrix': rectification_matrix.astype(np.float32),
                    'velodyne_to_cam_matrix': velodyne_to_cam_matrix.astype(np.float32)}
        else:
            raise FileNotFoundError('File %s does not exist' % calib_file_path)

    def _load_lidar(self, img_idx):
        velo_file_path = self.velo_files[img_idx]
        if exists(velo_file_path):
            point_cloud = np.fromfile(velo_file_path, np.single).reshape(-1, 4)
            return point_cloud
        else:
            raise FileNotFoundError('File %s does not exist' % velo_file_path)

    def _get_annotation(self, idx, _type='3d'):
        if _type == '3d':
            return self.annotations.loc[
                       self.annotations['idx'] == idx, ['x', 'y', 'z', 'l', 'w', 'h', 'ry']].values.astype(np.float32), \
                   self.annotations.loc[
                       self.annotations['idx'] == idx, ['class']].replace(self.cls_dict).values, self.annotations.loc[
                       self.annotations['idx'] == idx, ['class']].values
        elif _type == '2d':
            return self.annotations.loc[
                       self.annotations['idx'] == idx, ['minx', 'miny', 'maxx', 'maxy']].values.astype(np.float32), \
                   self.annotations.loc[
                       self.annotations['idx'] == idx, ['class']].replace(self.cls_dict).values, self.annotations.loc[
                       self.annotations['idx'] == idx, ['class']].values

    def _get_point_cloud(self, img_idx, img_size):
        calib_dict = self._load_calibration(img_idx)
        lidar_data = self._load_lidar(img_idx)
        tr_lidar = self._lidar_to_cam(lidar_data=lidar_data, calib_dict=calib_dict)
        tr_lidar = tr_lidar[tr_lidar[:, 2] > 0]
        pts = tr_lidar.T
        pts_in_img = self._project_lidar_to_img(pts, calib_dict)
        #

        image_filter = (pts_in_img[:, 0] > 0) & \
                       (pts_in_img[:, 0] < img_size[1]) & \
                       (pts_in_img[:, 1] > 0) & \
                       (pts_in_img[:, 1] < img_size[0])
        return tr_lidar[image_filter].T  # .astype(np.float32)

    def _get_ground_plane(self, img_idx):
        # plane_file = self.planes_dir + '/%06d.txt' % img_idx
        plane_file = self.plane_files[img_idx]
        plane = pd.read_csv(plane_file, skiprows=3, header=None, sep=' ').dropna(axis=1).to_numpy()[0]
        if plane[1] > 0:
            plane = -plane
        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        self.current_ground_plane = plane
        return plane

    @staticmethod
    def _lidar_to_cam(lidar_data, calib_dict):
        identity_matrix = np.identity(4)
        identity_matrix[:3, :3] = calib_dict['rectification_matrix']
        rect_matrix = identity_matrix.copy()
        identity_matrix[:3, :4] = calib_dict['velodyne_to_cam_matrix']
        velodyne_to_cam_matrix = identity_matrix
        ones = np.ones((lidar_data.shape[0], lidar_data.shape[1]))
        ones[:, :3] = lidar_data[:, :3]
        lidar_data = ones
        tr_lidar = np.dot(np.dot(rect_matrix, velodyne_to_cam_matrix), lidar_data.T)
        return tr_lidar[:3].T

    @staticmethod
    def _project_lidar_to_img(pts, calib_dict):
        camera_matrix = calib_dict['frame_calib_matrix'][2]
        padded_pts = np.ones((pts.shape[0] + 1, pts.shape[1]))
        padded_pts[:3, :] = pts
        pts_2d = np.dot(camera_matrix, padded_pts)
        pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
        pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
        pts_2d = np.delete(pts_2d, 2, 0)
        return pts_2d.T

    def _calculate_filtered_anchors_and_info(self, idx, anchors_2d, ground_plane, point_cloud, labels, label_cls):

        cached_file = self.caching_folder / (str(self.sample_names[idx]) + '.npy')


        filtered_anchors, _ = self._filter_anchors(anchors=anchors_2d, point_cloud=point_cloud,
                                                   ground_plane=ground_plane)

        ious, offsets, class_idx = self._extract_info_about_anchors(filtered_anchors, labels=labels,
                                                                    labels_cls=label_cls)


        mask, pos_mask = get_iou_mask(ious, self.rpn_neg_iou_range, self.rpn_pos_iou_range,
                                      num_of_elements=self.mini_batch_size)

        info_array = np.zeros((len(ious), 14))
        info_array[:, 0], info_array[:, 1:7], info_array[:, 7:-1], info_array[:,
                                                                   -1] = ious, filtered_anchors, offsets, mask  # , mask , info_array[:, -1]


        np.save(cached_file, info_array)

        return ious, filtered_anchors, offsets, mask  # , mask

    def _load_filtered_anchors_and_info(self, idx):

        cached_file = self.caching_folder / (str(self.sample_names[idx]) + '.npy')

        info_array = np.load(cached_file)
        ious, filtered_anchors, offsets, mask = info_array[:, 0], info_array[:, 1:7], info_array[:, 7:-1], \
                                                info_array[:, -1]
        # mask
        return ious, filtered_anchors, offsets, mask.astype(np.bool)  # , mask.astype(np.bool)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):

        image = self._load_image(idx)
        labels, label_cls_ids, label_cls = self._get_annotation(idx)

        point_cloud = self._get_point_cloud(idx, image.shape[:2])
        ground_plane = self._get_ground_plane(idx)
        calib_dict = self._load_calibration(idx)

        # anchor generation
        # no need to generate anchors using the flipped ground plane;just flip them after obtaining the filtered ones
        anchors_3 = self._generate_anchors(ground_plane=ground_plane)
        # anchors to axis aligned format
        anchors_2 = self._anchors_to_2d(anchors_3)
        # calibration
        # labels to axis aligned format

        # 	try to load filtered anchors alongside with their info if loading fails the _calculate...
        # 	is called and saves the results to a file for later usage
        try:
            # , mask
            ious, filtered_anchors, offsets, mask = self._load_filtered_anchors_and_info(idx)

        except FileNotFoundError:
            # , mask
            ious, filtered_anchors, offsets, mask = self._calculate_filtered_anchors_and_info(idx, anchors_2,
                                                                                              ground_plane, point_cloud,
                                                                                              labels, label_cls)

        res = self._transform_image(image=image, point_cloud=point_cloud, labels=labels,
                                    ground_plane=ground_plane, calib_dict=calib_dict,
                                    anchors=filtered_anchors, offsets=offsets)
        image, point_cloud, calib_dict, labels, ground_plane, filtered_anchors, offsets = res['image'], \
                                                                                          res['point_cloud'], \
                                                                                          res['calib_dict'], \
                                                                                          res['labels'], \
                                                                                          res['ground_plane'], \
                                                                                          res['anchors'], \
                                                                                          res['offsets']
        bev_maps = self.preprocess_point_cloud(point_cloud, ground_plane)

        anchors_gt = self._anchors_to_2d(labels, True)
        # axis aligned anchors are projected to BEV space
        anchors_gt_bev = project_to_bev(anchors_gt, self.area_extents, normalize=False)

        bev_anchors = project_to_bev(filtered_anchors, self.area_extents, normalize=True)
        rgb_anchors = project_axis_aligned_to_img(filtered_anchors, self.current_image_shape, calib_dict,
                                                  normalize=True)

        # CREATING ONE HOT TARGETS FOR RPN BASED ON MASK(IOU>MIN POS IOU)
        objectness = create_one_hot(ious >= self.rpn_pos_iou_range[0], neg_val=0.001)
        # if self.model == 'AVOD':
        return image, bev_maps, anchors_gt_bev, filtered_anchors, bev_anchors, rgb_anchors, ious, offsets, \
               objectness, mask, calib_dict, label_cls_ids, labels, ground_plane, self.current_image_shape, \
               self.sample_name


    def _load_bev_maps(self):
        return torch_load(self.bev_maps_folder / (self.sample_name + '.pt'))

    def _prepare_transforms(self):
        pass

    @staticmethod
    def _filter_annotation2_(annotations, difficulty):
        diff_dict = {'0': [40, 0, 0.15], '1': [25, 1, 0.3], '2': [25, 2, 0.5]}
        if isinstance(difficulty, list):
            # 		check for validity
            difficulty = set(difficulty) & {0, 1, 2}
            difficulty = str(max(difficulty))
            annotations = annotations[((annotations['maxy'] - annotations['miny']) >= diff_dict[difficulty][0]) &
                                      (annotations['occlusion'] <= diff_dict[difficulty][1]) &
                                      (annotations['truncation'] <= diff_dict[difficulty][2])]
            return annotations

