# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from fcn.config import cfg
import numpy as np

class Voxelizer(object):
    def __init__(self, grid_size, num_classes):
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.margin = 0.3
        self.min_x = 0
        self.min_y = 0
        self.min_z = 0
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        self.step_x = 0
        self.step_y = 0
        self.step_z = 0
        self.voxelized = False
        self.height = 0
        self.width = 0

    def setup(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

        # step size
        self.step_x = (max_x - min_x) / self.grid_size
        self.step_y = (max_y - min_y) / self.grid_size
        self.step_z = (max_z - min_z) / self.grid_size
        self.voxelized = True

    def draw(self, labels, colors, ax):

        for i in range(1, len(colors)):
            index = np.where(labels == i)
            X = index[0] * self.step_x + self.min_x
            Y = index[1] * self.step_y + self.min_y
            Z = index[2] * self.step_z + self.min_z
            ax.scatter(X, Y, Z, c=colors[i], marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        set_axes_equal(ax)

    def reset(self):
        self.min_x = 0
        self.min_y = 0
        self.min_z = 0
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        self.step_x = 0
        self.step_y = 0
        self.step_z = 0
        self.voxelized = False

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
            self.max_x = Xmax
            self.max_y = Ymax
            self.max_z = Zmax

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
        if cfg.FLIP_X:
            Kinv[0, 0] = -1 * Kinv[0, 0]
            Kinv[0, 2] = -1 * Kinv[0, 2]

        # compute the 3D points        
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

        # backprojection
        R = Kinv * x2d.transpose()

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

        # mask
        index = np.where(im_depth.flatten() == 0)
        X[:,index] = np.nan

        return np.array(X)

    def check_points(self, points, pose):
        # transform the points
        R = pose[0:3, 0:3]
        T = pose[0:3, 3].reshape((3,1))
        points = np.dot(R, points) + np.tile(T, (1, points.shape[1]))

        Xmin = np.nanmin(points[0,:])
        Xmax = np.nanmax(points[0,:])
        Ymin = np.nanmin(points[1,:])
        Ymax = np.nanmax(points[1,:])
        Zmin = np.nanmin(points[2,:])
        Zmax = np.nanmax(points[2,:])
        if Xmin >= self.min_x and Xmax <= self.max_x and Ymin >= self.min_y and Ymax <= self.max_y and Zmin >= self.min_z and Zmax <= self.max_z:
            return True
        else:
            print 'points x limit: {} {}'.format(Xmin, Xmax)
            print 'points y limit: {} {}'.format(Ymin, Ymax)
            print 'points z limit: {} {}'.format(Zmin, Zmax)
            return False

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
