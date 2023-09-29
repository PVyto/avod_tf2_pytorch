import numpy as np


class VoxelizePointCloud:
    # https://github.com/kujason/avod
    def __init__(self):
        # Quantization size of the voxel grid
        self.voxel_size = 0.0

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
        # Implementation taken from AVOD's official repository
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
        # unique non empty voxels
        coords = quantized_pts_2d[unique_pt_idxs]
        # find the contained points in each voxel
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
        #
        # self.min_voxel_coord = np.amin(coords, axis=0)
        # self.max_voxel_coord = np.amax(coords, axis=0)
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord) + 1).astype(np.int32)
        self.voxel_indices = (coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:
            self.leaf_layout_2d = -1 * np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout_2d[self.voxel_indices[:, 0], 0, self.voxel_indices[:, 2]] = 0

    def get_indices(self, corners):
        idxs = (corners / self.voxel_size).astype(np.int32) - self.min_voxel_coord[[0, 2]]
        return np.clip(idxs, [0, 0], self.num_divisions[[0, 2]])
