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

cdef extern from "synthesizer.hpp":
    cdef cppclass Synthesizer:
        Synthesizer(string, string) except +
        void setup(int, int )
        void render(int, int, float, float, float, float, float, float, unsigned char*, float*, float*, float*, float*, float*, float*, float*, float)
        void render_one(int, int, int, float, float, float, float, float, float, unsigned char*, float*, float*, float*, float*, float*)
        void solveICP(int*, unsigned char*, int, int, float, float, float, float, float, float, float, int, int, const float*, const float*, float*, float*, float)
        void estimatePose2D(int*, float*, float*, int, int, int, float, float, float, float, float*)
        void estimatePose3D(int*, unsigned char*, float*, float*, int, int, int, float, float, float, float, float, float*)

cdef class PySynthesizer:
    cdef Synthesizer *synthesizer     # hold a C++ instance which we're wrapping

    def __cinit__(self, string model_file, string pose_file):
        self.synthesizer = new Synthesizer(model_file, pose_file)

    def __dealloc__(self):
        del self.synthesizer

    def setup(self, int width, int height):
        self.synthesizer.setup(width, height)

    def render(self, np.ndarray[np.uint8_t, ndim=3] color, np.ndarray[np.float32_t, ndim=2] depth, np.ndarray[np.float32_t, ndim=3] vertmap, \
               np.ndarray[np.float32_t, ndim=1] class_indexes, np.ndarray[np.float32_t, ndim=2] poses, np.ndarray[np.float32_t, ndim=2] centers,\
               np.ndarray[np.float32_t, ndim=3] vertex_targets, np.ndarray[np.float32_t, ndim=3] vertex_weights, \
               np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py, np.float32_t znear, np.float32_t zfar, np.float32_t weight):

        cdef unsigned char* color_buff = <unsigned char*> color.data
        cdef int height = color.shape[0]
        cdef int width = color.shape[1]

        return self.synthesizer.render(width, height, fx, fy, px, py, znear, zfar, color_buff, &depth[0, 0], \
                   &vertmap[0, 0, 0], &class_indexes[0], &poses[0, 0], &centers[0, 0], &vertex_targets[0, 0, 0], &vertex_weights[0, 0, 0], weight)


    def render_one(self, np.ndarray[np.uint8_t, ndim=3] color, np.ndarray[np.float32_t, ndim=2] depth, np.ndarray[np.float32_t, ndim=3] vertmap, \
               np.ndarray[np.float32_t, ndim=2] poses, np.ndarray[np.float32_t, ndim=2] centers, np.ndarray[np.float32_t, ndim=2] extents, \
               float fx, float fy, float px, float py, float znear, float zfar, int which_class):

        cdef unsigned char* color_buff = <unsigned char*> color.data
        cdef int height = color.shape[0]
        cdef int width = color.shape[1]

        return self.synthesizer.render_one(which_class, width, height, fx, fy, px, py, znear, zfar, color_buff, &depth[0, 0], \
                   &vertmap[0, 0, 0], &poses[0, 0], &centers[0, 0], &extents[0, 0])


    def refine_poses(self, np.ndarray[np.int32_t, ndim=2] labels, np.ndarray[np.uint16_t, ndim=2] depth, np.ndarray[np.float32_t, ndim=2] rois, \
               np.ndarray[np.float32_t, ndim=2] poses, np.ndarray[np.float32_t, ndim=2] poses_new, np.ndarray[np.float32_t, ndim=2] poses_icp, \
               np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py, np.float32_t znear, np.float32_t zfar, np.float32_t factor, np.float32_t error):

        cdef int height = labels.shape[0]
        cdef int width = labels.shape[1]
        cdef int num_roi = rois.shape[0]
        cdef int channel_roi = rois.shape[1]
        cdef unsigned char* depth_buff = <unsigned char*> depth.data

        return self.synthesizer.solveICP(<int *>labels.data, depth_buff, height, width, fx, fy, px, py, znear, \
                                               zfar, factor, num_roi, channel_roi, &rois[0, 0], &poses[0, 0], \
                                               &poses_new[0, 0], &poses_icp[0, 0], error)


    def estimate_poses_2d(self, np.ndarray[np.int32_t, ndim=2] labels, \
               np.ndarray[np.float32_t, ndim=3] vertmap, np.ndarray[np.float32_t, ndim=2] extents, np.ndarray[np.float32_t, ndim=3] poses, \
               int num_classes, np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py):

        cdef int height = labels.shape[0]
        cdef int width = labels.shape[1]

        return self.synthesizer.estimatePose2D(<int *>labels.data, &vertmap[0, 0, 0], &extents[0, 0], \
            width, height, num_classes, fx, fy, px, py, &poses[0, 0, 0])


    def estimate_poses_3d(self, np.ndarray[np.int32_t, ndim=2] labels, np.ndarray[np.uint16_t, ndim=2] depth, \
               np.ndarray[np.float32_t, ndim=3] vertmap, np.ndarray[np.float32_t, ndim=2] extents, np.ndarray[np.float32_t, ndim=3] poses, \
               int num_classes, np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py, np.float32_t factor):

        cdef int height = labels.shape[0]
        cdef int width = labels.shape[1]
        cdef unsigned char* depth_buff = <unsigned char*> depth.data

        return self.synthesizer.estimatePose3D(<int *>labels.data, depth_buff, &vertmap[0, 0, 0], &extents[0, 0], \
            width, height, num_classes, fx, fy, px, py, factor, &poses[0, 0, 0])
