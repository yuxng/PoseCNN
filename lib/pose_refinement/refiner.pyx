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

cdef extern from "refiner.hpp":
    cdef cppclass Refiner:
        Refiner(string) except +
        void setup(string)
        void render(unsigned char*, unsigned char*, float*, int, int, int, int, int, float*, float*, float, float, float, float, float*, float*, int)

cdef class PyRefiner:
    cdef Refiner *refiner     # hold a C++ instance which we're wrapping

    def __cinit__(self, string model_file):
        self.refiner = new Refiner(model_file)

    def __dealloc__(self):
        del self.refiner

    def setup(self, string filename):
        return self.refiner.setup(filename)

    def render(self, np.ndarray[np.uint8_t, ndim=3] color, np.ndarray[np.uint8_t, ndim=2] label, np.ndarray[np.float32_t, ndim=2] rois, \
               np.ndarray[np.float32_t, ndim=2] poses_gt, np.ndarray[np.float32_t, ndim=2] poses_pred, np.float32_t fx, \
               np.float32_t fy, np.float32_t px, np.float32_t py, int num_classes, np.ndarray[np.float32_t, ndim=2] extents, np.ndarray[np.float32_t, ndim=2] poses_new, int is_save):

        cdef unsigned char* color_buff = <unsigned char*> color.data
        cdef unsigned char* label_buff = <unsigned char*> label.data
        cdef int height = color.shape[0]
        cdef int width = color.shape[1]
        cdef int num_rois = rois.shape[0]
        cdef int num_gt = poses_gt.shape[0]

        return self.refiner.render(color_buff, label_buff, &rois[0, 0], num_rois, num_gt, width, height, num_classes, \
                                   &poses_gt[0, 0], &poses_pred[0, 0], fx, fy, px, py, &extents[0, 0], &poses_new[0, 0], is_save)
