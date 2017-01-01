#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import gpu_normals
import os
import scipy.io

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

if __name__ == '__main__':

    root_dir = '/var/Projects/FCN/data/RGBDScene/data'

    fx = 570.3   # Focal length in x
    fy = 570.3   # Focal length in x
    cx = 320.0   # Center of projection in x
    cy = 240.0   # Center of projection in y
    depthCutoff = 20.0
    nmaps = []

    for i in range(14):
        filename = os.path.join(root_dir, 'scene_{:02d}'.format(i+1), '00000-depth.png')
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        depth = im.astype(np.float32, copy=True) / 10000.0

        nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, depthCutoff, 0)
        print nmap.shape, np.nanmin(nmap), np.nanmax(nmap)
        nmaps.append(nmap)

        '''
        # convert normals to an image
        N = 127.5*nmap + 127.5
        N = N.astype(np.uint8)

        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(im)

        fig.add_subplot(122)
        plt.imshow(N)
        plt.show()
        '''

    # save results
    nmaps = {'nmaps': nmaps}
    scipy.io.savemat('nmaps.mat', nmaps)
