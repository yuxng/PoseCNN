# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np

class Voxelizer(object):
    def __init__(self, grid_size, num_classes):
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.margin = 0.1
        self.filter_h = 5
        self.filter_w = 5
        self.min_x = 0
        self.min_y = 0
        self.min_z = 0
        self.step_x = 0
        self.step_y = 0
        self.step_z = 0
        self.voxelized = False
        self.height = 0
        self.width = 0
        self.data = np.zeros((grid_size, grid_size, grid_size, num_classes), dtype=np.float32)
        self.count = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)

    def update(self, im, grid_indexes):
        height = im.shape[0]
        width = im.shape[1]
        assert height*width == grid_indexes.shape[1], \
               'in voxelizer.update, image shape {} not compatible to grid indexes {}'.format(height*width, grid_indexes.shape[1])
        
        index = np.where(im.flatten() > 0)[0]
        for i in range(len(index)):
            ind = index[i]
            gx = grid_indexes[0, ind]
            gy = grid_indexes[1, ind]
            gz = grid_indexes[2, ind]
            if np.isfinite(gx) and np.isfinite(gy) and np.isfinite(gz):
                if gx >= 0 and gx < self.grid_size and gy >= 0 and gy < self.grid_size and gz >= 0 and gz < self.grid_size:
                    self.data[int(gx), int(gy), int(gz), :] = 1
                    self.count[int(gx), int(gy), int(gz)] += 1

    def draw(self):
        index = np.where(self.count > 0)
        X = index[0] * self.step_x + self.min_x
        Y = index[1] * self.step_y + self.min_y
        Z = index[2] * self.step_z + self.min_z
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)
        plt.show()

    def reset(self):
        self.min_x = 0
        self.min_y = 0
        self.min_z = 0
        self.step_x = 0
        self.step_y = 0
        self.step_z = 0
        self.voxelized = False
        self.data = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.num_classes), dtype=np.float32)
        self.count = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int32)

    def voxelize(self, points):
        if not self.voxelized:
            # compute the boundary of the 3D points
            Xmin = np.nanmin(points[0,:]) - self.margin
            Xmax = np.nanmax(points[0,:]) + self.margin
            Ymin = np.nanmin(points[1,:]) - self.margin
            Ymax = np.nanmax(points[1,:]) + self.margin
            Zmin = np.nanmin(points[2,:]) - self.margin
            Zmax = np.nanmax(points[2,:]) + self.margin
            self.min_x = Xmin
            self.min_y = Ymin
            self.min_z = Zmin

            # step size
            self.step_x = (Xmax-Xmin) / self.grid_size
            self.step_y = (Ymax-Ymin) / self.grid_size
            self.step_z = (Zmax-Zmin) / self.grid_size
            self.voxelized = True

        # compute grid indexes
        indexes = np.zeros_like(points, dtype=np.float32)
        indexes[0,:] = np.floor((points[0,:] - self.min_x) / self.step_x)
        indexes[1,:] = np.floor((points[1,:] - self.min_y) / self.step_y)
        indexes[2,:] = np.floor((points[2,:] - self.min_z) / self.step_z)

        # crash the grid indexes
        # grid_indexes = indexes[0,:] * self.grid_size * self.grid_size + indexes[1,:] * self.grid_size + indexes[2,:]
        # I = np.isnan(grid_indexes)
        # grid_indexes[I] = -1
        # grid_indexes = grid_indexes.reshape(self.height, self.width).astype(np.int32)

        return indexes

    # backproject pixels into 3D points
    def backproject(self, im_depth, meta_data):
        depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

        # compute projection matrix
        P = meta_data['projection_matrix']
        P = np.matrix(P)
        Pinv = np.linalg.pinv(P)

        # compute the 3D points        
        height = depth.shape[0]
        width = depth.shape[1]
        self.height = height
        self.width = width

        # camera location
        C = meta_data['camera_location']
        C = np.matrix(C).transpose()
        Cmat = np.tile(C, (1, width*height))

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        x3d = Pinv * x2d.transpose()
        x3d[0,:] = x3d[0,:] / x3d[3,:]
        x3d[1,:] = x3d[1,:] / x3d[3,:]
        x3d[2,:] = x3d[2,:] / x3d[3,:]
        x3d = x3d[:3,:]

        # compute the ray
        R = x3d - Cmat

        # compute the norm
        N = np.linalg.norm(R, axis=0)
        
        # normalization
        R = np.divide(R, np.tile(N, (3,1)))

        # compute the 3D points
        X = Cmat + np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

        # mask
        index = np.where(im_depth.flatten() == 0)
        X[:,index] = np.nan

        return np.array(X)

    # backproject pixels into 3D points in camera's coordinate system
    def backproject_camera(self, im_depth, meta_data):

        depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

        # get intrinsic matrix
        K = meta_data['intrinsic_matrix']
        K = np.matrix(K)
        Kinv = np.linalg.inv(K)

        # compute the 3D points        
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        R = Kinv * x2d.transpose()

        # compute the norm
        N = np.linalg.norm(R, axis=0)
        
        # normalization
        R = np.divide(R, np.tile(N, (3,1)))

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

        # mask
        index = np.where(im_depth.flatten() == 0)
        X[:,index] = np.nan

        return np.array(X)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
