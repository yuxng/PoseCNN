# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

from libcpp.string cimport string
import numpy as np
cimport numpy as np
import ctypes

cdef extern from "kfusion.hpp" namespace "df":
    cdef cppclass KinectFusion:
        KinectFusion(string) except +
        void solve_pose(float*, float*)
        void fuse_depth()
        void extract_surface(int*)
        void render()
        void draw(string, int)
        void back_project()
        void feed_data(unsigned char*, unsigned char*, int, int, float)
        void feed_label(unsigned char*, float*, unsigned char*)
        void reset()
        void set_voxel_grid(float, float, float, float, float, float);
        void save_model(string)

cdef class PyKinectFusion:
    cdef KinectFusion *kfusion     # hold a C++ instance which we're wrapping

    def __cinit__(self, string rig_file):
        self.kfusion = new KinectFusion(rig_file)

    def __dealloc__(self):
        del self.kfusion

    def solve_pose(self):
        cdef np.ndarray[np.float32_t, ndim=2] pose_world2live = np.zeros((3, 4), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2] pose_live2world = np.zeros((3, 4), dtype=np.float32)
        self.kfusion.solve_pose(&pose_world2live[0, 0], &pose_live2world[0, 0])
        return pose_world2live, pose_live2world

    def fuse_depth(self):
        return self.kfusion.fuse_depth()

    def extract_surface(self, np.ndarray[np.int32_t, ndim=2] labels):
        cdef np.ndarray[int, ndim=2, mode="c"] l_c
        l_c = np.ascontiguousarray(labels, dtype=ctypes.c_int)
        self.kfusion.extract_surface(&l_c[0, 0])
        return l_c

    def render(self):
        return self.kfusion.render()

    def draw(self, string filename, np.int32_t flag):
        return self.kfusion.draw(filename, flag)

    def back_project(self):
        return self.kfusion.back_project()

    def feed_data(self, np.ndarray[np.uint16_t, ndim=2] depth, np.ndarray[np.uint8_t, ndim=3] color, np.int32_t width, np.int32_t height, np.float32_t factor):
        cdef unsigned char* depth_buff = <unsigned char*> depth.data
        cdef unsigned char* color_buff = <unsigned char*> color.data
        return self.kfusion.feed_data(depth_buff, color_buff, width, height, factor)

    def feed_label(self, np.ndarray[np.uint8_t, ndim=3] im_label, np.ndarray[np.float32_t, ndim=3] probs, np.ndarray[np.uint8_t, ndim=1] colors):
        cdef unsigned char* im_label_buff = <unsigned char*> im_label.data
        cdef unsigned char* colors_buff = <unsigned char*> colors.data
        return self.kfusion.feed_label(im_label_buff, &probs[0, 0, 0], colors_buff)

    def reset(self):
        return self.kfusion.reset()

    def set_voxel_grid(self, float voxelGridOffsetX, float voxelGridOffsetY, float voxelGridOffsetZ, float voxelGridDimX, float voxelGridDimY, float voxelGridDimZ):
        return self.kfusion.set_voxel_grid(voxelGridOffsetX, voxelGridOffsetY, voxelGridOffsetZ, voxelGridDimX, voxelGridDimY, voxelGridDimZ)

    def save_model(self, string filename):
        return self.kfusion.save_model(filename)
