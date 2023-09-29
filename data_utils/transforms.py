import torch
import torchvision
import numpy as np
import tensorflow as tf
from general_utils import export
from skimage.transform import resize
from torchvision.transforms import ColorJitter
from data_utils.dataset_utils import create_point_filter
from abc import abstractmethod

__all__ = []


def plot_before_after(fn):
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        # plt.rcParams['axes.xmargin'] = 0
        # plt.margins(x=0)
        fig, ax = plt.subplots(2, figsize=(15, 15))

        image_b = kwargs['image']
        tr_name = args[0].__class__.__name__
        ax[0].title.set_text('image before {} transform'.format(tr_name))
        ax[0].imshow(image_b.astype(np.uint8))
        # image_b
        k = fn(*args, **kwargs)
        image = k['image']
        if image.max() <= 1.:
            image = image * 255.
        image = image.astype(np.uint8)
        ax[1].title.set_text('image after {} transform'.format(tr_name))
        ax[1].imshow(image)
        fig.tight_layout()
        plt.show()
        return k

    return wrapper


def _blend(img1, img2, ratio):
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def _adjust_brightness(img, brightness_factor):
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    # _assert_image_tensor(img)

    # _assert_channels(img, [1, 3])

    return _blend(img, torch.zeros_like(img), brightness_factor)


def plot_image(image):
    import matplotlib.pyplot as plt
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if image.max() <= 1.:
        image *= 255.
    print(np.unique(image))
    plt.imshow(image.astype(np.uint8))
    plt.show()


def adjust_brightness(image, factor=1.):
    import torchvision.transforms.functional as f
    if isinstance(image, np.ndarray):
        image = torch.as_tensor(image).permute(2, 0, 1)
    print(np.unique(image))
    image = f.adjust_brightness(image, brightness_factor=factor)
    return image


# https://github.com/kujason/avod
@export
class VoxelizePointCloud:

    def __init__(self):
        # Quantization size of the voxel grid
        self.voxel_size = 1.0

        # Voxels at the most negative/positive xyz
        self.min_voxel_coord = np.array([])
        self.max_voxel_coord = np.array([])

        # Size of the voxel grid along each axis
        self.num_divisions = np.array([0, 0, 0])

        # Points in sorted order, to match the order of the voxels
        self.points = []

        # Indices of filled voxels
        self.voxel_indices = []

        # Max point height in projected voxel
        self.heights = []

        # Number of points corresponding to projected voxel
        self.num_pts_in_voxel = []

    def create_voxel_grid(self, voxel_size, point_cloud, ground_plane, extents, create_leaf_layout=False):
        self.voxel_size = voxel_size
        quantized_pts = np.floor(point_cloud / voxel_size).astype(np.int32)
        x_dim, y_dim, z_dim = quantized_pts.T[[0, 1, 2]]
        sorted_idx = np.lexsort((y_dim, z_dim, x_dim))
        self.points = point_cloud[sorted_idx]
        quantized_pts = quantized_pts[sorted_idx]
        quantized_pts_2d = quantized_pts.copy()
        quantized_pts_2d[:, 1] = 0
        cont_array = np.ascontiguousarray(quantized_pts_2d).view(
            np.dtype((np.void, quantized_pts_2d.dtype.itemsize * quantized_pts_2d.shape[1])))
        _, unique_pt_idxs = np.unique(cont_array, return_index=True)
        unique_pt_idxs.sort()
        coords = quantized_pts_2d[unique_pt_idxs]
        num_of_points_in_voxel = np.diff(unique_pt_idxs)
        num_of_points_in_voxel = np.append(num_of_points_in_voxel, quantized_pts_2d.shape[0] - unique_pt_idxs[-1])

        # height_in_voxel = self.points[unique_pt_idxs, 1]
        height_in_voxel = (np.dot(self.points[unique_pt_idxs], ground_plane[:3]) + ground_plane[3]) / (
            np.sqrt(np.sum(ground_plane[:3] ** 2)))
        self.heights = height_in_voxel
        self.num_pts_in_voxel = num_of_points_in_voxel
        self.min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        self.max_voxel_coord = np.ceil((extents.T[1] / voxel_size) - 1)

        self.min_voxel_coord[1] = 0
        self.max_voxel_coord[1] = 0
        
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord) + 1).astype(np.int32)
        self.voxel_indices = (coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:
            self.leaf_layout_2d = -1 * np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout_2d[self.voxel_indices[:, 0], 0, self.voxel_indices[:, 2]] = 0

    def get_indices(self, corners):
        idxs = (corners / self.voxel_size).astype(np.int32) - self.min_voxel_coord[[0, 2]]
        return np.clip(idxs, [0, 0], self.num_divisions[[0, 2]])


class BevSlicesCreator:
    _config_dict = {'extents': [-40.0, 40.0, -5.0, 3.0, 0.0, 70.0], 'voxel_size': 0.10000000149011612,
                    'height_lo': -0.20000000298023224, 'height_hi': 2.299999952316284, 'num_slices': 5}

    def __init__(self, config=None):
        if config is None:
            config = self._config_dict
        self.config = config
        self.extents = np.array(config['area_extents']).reshape(3, 2)
        self.height_lo = config['height_lo']
        self.height_hi = config['height_hi']
        self.num_slices = config['num_slices']
        self.voxel_size = config['voxel_size']
        self.height_per_division = (self.height_hi - self.height_lo) / self.num_slices
        self.heights = [(self.height_lo + i * self.height_per_division,
                         self.height_lo + i * self.height_per_division + self.height_per_division)
                        for i in range(self.num_slices)]
        self.density_norm_value = np.log(16)

    def __call__(self, point_cloud, ground_plane):
        point_cloud_T = point_cloud.T
        height_maps = []
        for sidx in range(self.num_slices):
            # calculate the min and max height for each slice
            slice_filter = self.create_slice_filter(point_cloud, ground_plane, sidx)

            points = point_cloud_T[slice_filter]
            voxel_grid = VoxelizePointCloud()
            voxel_grid.create_voxel_grid(self.voxel_size, points, ground_plane, self.extents)

            height_map = self.create_height_map(voxel_grid=voxel_grid, slice_index=sidx)

            height_maps.append(height_map)

        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0) for map_idx in range(len(height_maps))]

        density_filter = self.create_slice_filter(point_cloud, ground_plane)

        density_points = point_cloud_T[density_filter]

        # Create Voxel Grid 2D
        density_voxel_grid_2d = VoxelizePointCloud()
        density_voxel_grid_2d.create_voxel_grid(self.voxel_size, density_points, ground_plane=ground_plane,
                                                extents=self.extents)  # density_voxel_grid_2d

        # Generate density map
        density_voxel_indices_2d = density_voxel_grid_2d.voxel_indices[:, [0, 2]]

        density_map = self.create_density_map(
            num_divisions=density_voxel_grid_2d.num_divisions,
            voxel_indices=density_voxel_indices_2d,
            pts_per_vox=density_voxel_grid_2d.num_pts_in_voxel,
        )

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['density_map'] = density_map

        return bev_maps

    def create_slice_filter(self, point_cloud, ground_plane, slice_index=None):
        height_l, height_h = self.height_lo, self.height_hi
        if slice_index is not None:
            height_l, height_h = self.heights[slice_index]
        offset_filter = create_point_filter(self.extents, point_cloud=point_cloud, ground_plane=ground_plane,
                                            offset=height_h)
        road_filter = create_point_filter(self.extents, point_cloud=point_cloud, ground_plane=ground_plane,
                                          offset=height_l)
        return np.logical_xor(offset_filter, road_filter)

    def create_height_map(self, voxel_grid, slice_index):
        h_map = np.zeros((voxel_grid.num_divisions[0], voxel_grid.num_divisions[2]))
        voxel_grid.heights -= self.heights[slice_index][0]
        voxel_idxs = voxel_grid.voxel_indices[:, [0, 2]]
        h_map[voxel_idxs[:, 0], voxel_idxs[:, 1]] = np.array(voxel_grid.heights) / self.height_per_division
        return h_map

    def create_density_map(self, num_divisions, voxel_indices, pts_per_vox, val_to_norm=None):
        if val_to_norm is None:
            val_to_norm = self.density_norm_value
        dens_map = np.zeros((num_divisions[0], num_divisions[2]))
        dens_map[voxel_indices[:, 0], voxel_indices[:, 1]] = \
            np.minimum(1.0, np.log(pts_per_vox + 1) / val_to_norm)

        # Density is calculated as min(1.0, log(N+1)/log(x))
        dens_map = np.flip(dens_map.T, axis=0)

        return dens_map


@export
class ToTensor:
    def __call__(self, normalize=False, **kwargs):
        image = kwargs.get('image')
        img = image.transpose(2, 0, 1)
        if normalize:
            img = (img / 255.0).astype(np.float32)
        # copy to avoid a ValueError exception
        kwargs.update(image=torch.as_tensor(img.copy(), dtype=torch.float32))
        return kwargs



@export
class ToTfTensor:
    def __call__(self, normalize=False, **kwargs):
        image = kwargs.get('image')
        img = image
        if normalize:
            img = (img / 255.0).astype(np.float32)
        # copy to avoid a ValueError exception
        kwargs.update(image=tf.convert_to_tensor(img.copy(), dtype=tf.float32))
        return kwargs


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        # self.multiple_sources = multiple_sources

    def __call__(self, **input_dict):
        res = input_dict
        for transform in self.transforms:
            res = transform(**res)
        return res


class BaseKittiPCTransform:
    def __init__(self, config):
        self.bev_creator = BevSlicesCreator(config)

    @abstractmethod
    def __call__(self, point_cloud, ground_plane):
        pass


class KittiPCTransformTorch(BaseKittiPCTransform):

    def __call__(self, point_cloud, ground_plane):
        # pad bev maps for the FPN to work properly;remove the padding afterwards(FeatureExtractor's forward method)
        bev_maps = self.bev_creator(point_cloud=point_cloud, ground_plane=ground_plane)
        height_tensors = [torch.as_tensor(np.pad(m.copy(), ((4, 0), (0, 0)))) for m in bev_maps['height_maps']]
        dens_tensor = torch.as_tensor(np.pad(bev_maps['density_map'].copy(), ((4, 0), (0, 0))))
        bev_tensors = torch.stack(height_tensors + [dens_tensor])
        return bev_tensors.float()


class KittiPCTransformTf(BaseKittiPCTransform):

    def __call__(self, point_cloud, ground_plane):
        # pad bev maps for the FPN to work properly;remove the padding afterwards(FeatureExtractor's forward method)
        bev_maps = self.bev_creator(point_cloud=point_cloud, ground_plane=ground_plane)
        height_tensors = [tf.convert_to_tensor(np.pad(m.copy(), ((4, 0), (0, 0)))) for m in bev_maps['height_maps']]
        dens_tensor = tf.convert_to_tensor(np.pad(bev_maps['density_map'].copy(), ((4, 0), (0, 0))))
        bev_tensors = tf.stack(height_tensors + [dens_tensor], axis=2)
        return bev_tensors


class BaseKittiPCTransform:
    def __init__(self, config):
        self.bev_creator = BevSlicesCreator(config)

    def __call__(self, point_cloud, ground_plane):
        # pad bev maps for the FPN to work properly;remove the padding afterwards(FeatureExtractor's forward method)
        bev_maps = self.bev_creator(point_cloud=point_cloud, ground_plane=ground_plane)
        height_tensors = [torch.as_tensor(np.pad(m.copy(), ((4, 0), (0, 0)))) for m in bev_maps['height_maps']]
        dens_tensor = torch.as_tensor(np.pad(bev_maps['density_map'].copy(), ((4, 0), (0, 0))))
        bev_tensors = torch.stack(height_tensors + [dens_tensor])
        return bev_tensors.float()


class BaseKittiImageTransform:
    def __init__(self):
        self._image_transform = Compose([ResizeTransform(), MeanNormalize(), ToTensor()])

    def __call__(self, **kwargs):
        return self._image_transform(**kwargs)


class BaseKittiImageTransformTf:
    def __init__(self):
        self._image_transform = Compose([ResizeTransform(), MeanNormalize(), ToTfTensor()])

    def __call__(self, **kwargs):
        return self._image_transform(**kwargs)


@export
class MeanNormalize:
    def __init__(self):
        self.red_mean = 92.8403
        self.green_mean = 97.7996
        self.blue_mean = 93.5843

    def __call__(self, *args, **kwargs):
        image = kwargs.get('image')
        kwargs['image'] = image - [self.red_mean, self.green_mean, self.blue_mean]
        return kwargs


@export
class ResizeTransform:
    def __init__(self, shapes=None):
        if shapes is None:
            shapes = {'image': [360, 1200]}
        self.shapes = shapes

    def __call__(self, *args, **kwargs):
        for k, v in self.shapes.items():
            transformed = resize(kwargs.get(k).astype(np.float32), output_shape=v, order=1)
            kwargs[k] = transformed
        return kwargs


class RandomApplyTransform:
    def __init__(self, apply_probability=.5):
        # probability of transforming the image
        assert (apply_probability >= 0.0) and (apply_probability <= 1)
        self.probability_to_apply = apply_probability

    def call(self, *args, **kwargs):
        pass

    def __call__(self, **kwargs):
        apply = np.random.choice([False, True], p=[1 - self.probability_to_apply, self.probability_to_apply])
        if apply:
            return self.call(**kwargs)
        return kwargs


class AlwaysApplyTransform:
    def call(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.call(**kwargs)


@export
class ColorJitterTransform(RandomApplyTransform):
    def __init__(self):
        v = .4
        self.transform = ColorJitter(v, v, v)
        super(ColorJitterTransform, self).__init__()

    def call(self, **kwargs):
        image = kwargs.get('image')
        if isinstance(image, np.ndarray):
            image = torch.as_tensor(image.transpose([2, 0, 1]).copy().astype(np.uint8))

        kwargs.update(image=self.transform(image).permute(1, 2, 0).numpy().astype(np.float32))
        return kwargs


@export
class PCAJitterTransform(RandomApplyTransform):

    def call(self, **kwargs):
        image = kwargs.get('image')
        reshaped = image.reshape(-1, 3)
        reshaped = reshaped / 255.0
        covariance = np.cov(reshaped.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        pca = np.sqrt(eigenvalues) * eigenvectors
        transformed_img = image / 255.0
        add_vals = (pca * np.random.randn(3) * 0.1).sum(axis=1)
        transformed_img += add_vals
        np.clip(transformed_img, 0.0, 1.0, out=transformed_img)
        transformed_img = (transformed_img * 255).astype(np.uint8)
        kwargs.update(image=transformed_img)
        return kwargs


@export
class FlipTransform(RandomApplyTransform):

    def call(self, **kwargs):
        image = kwargs.get('image')
        labels = kwargs.get('labels')
        anchors = kwargs.get('anchors')
        offsets = kwargs.get('offsets')
        calib_dict = kwargs.get('calib_dict')
        point_cloud = kwargs.get('point_cloud')
        ground_plane = kwargs.get('ground_plane')

        h, w, _ = image.shape
        image = np.fliplr(image)
        # point cloud
        point_cloud[0] = -point_cloud[0]
        # ground plane
        ground_plane[0] = -ground_plane[0]
        # labels :[x,y,z,h,w,l,ry]
        # x
        # labels
        labels[:, 0] = -labels[:, 0]
        # ry
        labels[labels[:, -1] < 0, -1] = -np.pi - labels[labels[:, -1] < 0, -1]
        labels[labels[:, -1] >= 0, -1] = np.pi - labels[labels[:, -1] >= 0, -1]
        # calibration matrix
        p2_mat = calib_dict['frame_calib_matrix'][2]
        p2_mat[0, 3] = -p2_mat[0, 3]
        p2_mat[0, 2] = w - p2_mat[0, 2]
        calib_dict['frame_calib_matrix'][2] = p2_mat
        # anchors (axis aligned format)
        # correct y wrt the changed ground plane
        anchors[:, 1] = -(np.dot(ground_plane[[0, 2]], anchors[:, [0, 2]].T) + ground_plane[3]) / ground_plane[1]
        anchors[:, 0] = -anchors[:, 0]
        # offsets
        offsets[:, 0] = -offsets[:, 0]

        kwargs.update(image=image, point_cloud=point_cloud, labels=labels, calib_dict=calib_dict,
                      ground_plane=ground_plane, anchors=anchors, offsets=offsets)
        return kwargs


@export
class Unormalize(MeanNormalize):
    def __call__(self, *args, **kwargs):
        image = kwargs.get('image')
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        kwargs['image'] = image + [self.red_mean, self.green_mean, self.blue_mean]
        return kwargs


@export
class UnormalizeTf(MeanNormalize):
    def __call__(self, *args, **kwargs):
        image = kwargs.get('image')
        if tf.is_tensor(image):
            image = image.numpy()
        kwargs['image'] = image + [self.red_mean, self.green_mean, self.blue_mean]
        return kwargs

