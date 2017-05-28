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

cdef extern from "ransac.hpp" namespace "jp":
    cdef cppclass Ransac3D:
        Ransac3D() except +
        void estimatePose(unsigned char*, float*, float*, float*, int, int, int, float, float, float, float, float, float*)
        void estimateCenter(float*, float*, int, int, int, float*)

cdef class PyRansac3D:
    cdef Ransac3D *ransac3d     # hold a C++ instance which we're wrapping

    def __cinit__(self):
        self.ransac3d = new Ransac3D()

    def __dealloc__(self):
        del self.ransac3d

    def estimate_pose(self, np.ndarray[np.uint16_t, ndim=2] depth, np.ndarray[np.float32_t, ndim=3] probs, np.ndarray[np.float32_t, ndim=3] vertexs, \
        np.ndarray[np.float32_t, ndim=2] extents, np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py, np.float32_t depth_factor):

        cdef unsigned char* depth_buff = <unsigned char*> depth.data
        cdef int height = probs.shape[0]
        cdef int width = probs.shape[1]
        cdef int num_classes = probs.shape[2]

        cdef np.ndarray[np.float32_t, ndim=3] poses = np.inf * np.ones((3, 4, num_classes), dtype=np.float32)

        self.ransac3d.estimatePose(depth_buff, &probs[0, 0, 0], &vertexs[0, 0, 0], &extents[0, 0], width, height, num_classes, fx, fy, px, py, depth_factor, &poses[0, 0, 0])

        return poses

    def estimate_center(self, np.ndarray[np.float32_t, ndim=3] probs, np.ndarray[np.float32_t, ndim=3] vertexs):

        cdef int height = probs.shape[0]
        cdef int width = probs.shape[1]
        cdef int num_classes = probs.shape[2]

        cdef np.ndarray[np.float32_t, ndim=2] centers = np.inf * np.ones((num_classes, 4), dtype=np.float32)

        self.ransac3d.estimateCenter(&probs[0, 0, 0], &vertexs[0, 0, 0], width, height, num_classes, &centers[0, 0])

        return centers
