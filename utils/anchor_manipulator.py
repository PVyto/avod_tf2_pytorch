import numpy as np
from data_utils.dataset_utils import filter_point_cloud
from skimage.transform.integral import integral_image
from data_utils.transforms import VoxelizePointCloud


class AnchorManipulator:
    def __init__(self, stride, area_extents, density_threshold=1):
        self.stride = stride
        self.area_extents = area_extents
        self.density_threshold = density_threshold

    def generate_anchors_3d(self, ground_plane, anchor_sizes):
        x_extents, y_extents, z_extents = np.array(self.area_extents)

        x = np.arange(*(x_extents + np.array([self.stride[0] / 2, 0])), self.stride[0])
        z = np.arange(*(z_extents + np.array([self.stride[1] / 2, 0])), self.stride[1])[::-1]
        rotations = np.array([0, np.pi / 2])

        anchors_3d_x_z = np.stack(
            [i.reshape(-1) for i in np.meshgrid(x, z, np.arange(0, len(anchor_sizes)), np.arange(0, len(rotations)))],
            axis=1)
        anchors_3d = np.zeros((len(anchors_3d_x_z), 7))
        a, b, c, d = ground_plane
        anchors_3d[:, [0, 2]] = anchors_3d_x_z[:, [0, 1]]  # x, z centers
        anchors_3d[:, 1] = - (a * anchors_3d_x_z[:, 0] + c * anchors_3d_x_z[:, 1] + d) / b  # y centers
        anchors_3d[:, -1] = rotations[anchors_3d_x_z[:, -1].astype(int)]
        anchors_3d[:, 3:6] = anchor_sizes[anchors_3d_x_z[:, -2].astype(int)]

        return anchors_3d

    def get_non_empty_anchors_filter(self, anchors, point_cloud, ground_plane, voxel_size):
        pcf = filter_point_cloud(self.area_extents, point_cloud=point_cloud, ground_plane=ground_plane)
        voxel_ = VoxelizePointCloud()
        voxel_.create_voxel_grid(voxel_size, point_cloud.T[pcf], ground_plane, self.area_extents, True)

        integral_img = np.pad(integral_image((voxel_.leaf_layout_2d + 1).squeeze()), (1, 0))
        corners = np.zeros((anchors.shape[0], 4))
        corners[:, [0, 1]] = anchors[:, [0, 2]] - anchors[:, [3, 5]] / 2
        corners[:, [2, 3]] = anchors[:, [0, 2]] + anchors[:, [3, 5]] / 2
        anchor_indices = np.stack((voxel_.get_indices(corners[:, :2]), voxel_.get_indices(corners[:, 2:])),
                                  axis=1).reshape(-1, 4).astype(np.uint32)

        x1 = anchor_indices.T[0, :]
        z1 = anchor_indices.T[1, :]
        x2 = anchor_indices.T[2, :]
        z2 = anchor_indices.T[3, :]
        t = integral_img[x2, z2] + integral_img[x1, z1] - integral_img[x2, z1] - integral_img[x1, z2]
        return (t) >= self.density_threshold
