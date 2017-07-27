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
        void setup()
        void render(int, int, float, float, float, float, float, float, unsigned char*, float*, float*, float*, float*, float*, float*, float*, float)
        void estimateCenter(int*, float*, float*, int, int, int, int, float, float, float, float, float*)

cdef class PySynthesizer:
    cdef Synthesizer *synthesizer     # hold a C++ instance which we're wrapping

    def __cinit__(self, string model_file, string pose_file):
        self.synthesizer = new Synthesizer(model_file, pose_file)

    def __dealloc__(self):
        del self.synthesizer

    def setup(self):
        self.synthesizer.setup()

    def render(self, np.ndarray[np.uint8_t, ndim=3] color, np.ndarray[np.float32_t, ndim=2] depth, np.ndarray[np.float32_t, ndim=3] vertmap, \
               np.ndarray[np.float32_t, ndim=1] class_indexes, np.ndarray[np.float32_t, ndim=2] poses, np.ndarray[np.float32_t, ndim=2] centers,\
               np.ndarray[np.float32_t, ndim=3] vertex_targets, np.ndarray[np.float32_t, ndim=3] vertex_weights, \
               np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py, np.float32_t znear, np.float32_t zfar, np.float32_t weight):

        cdef unsigned char* color_buff = <unsigned char*> color.data
        cdef int height = color.shape[0]
        cdef int width = color.shape[1]

        return self.synthesizer.render(width, height, fx, fy, px, py, znear, zfar, color_buff, &depth[0, 0], \
                   &vertmap[0, 0, 0], &class_indexes[0], &poses[0, 0], &centers[0, 0], &vertex_targets[0, 0, 0], &vertex_weights[0, 0, 0], weight)


    def estimate_centers(self, np.ndarray[np.int32_t, ndim=2] labels, np.ndarray[np.float32_t, ndim=3] vertmap, \
               np.ndarray[np.float32_t, ndim=2] extents, np.ndarray[np.float32_t, ndim=2] centers, \
               int num_classes, int preemptive_batch,
               np.float32_t fx, np.float32_t fy, np.float32_t px, np.float32_t py):

        cdef int height = labels.shape[0]
        cdef int width = labels.shape[1]

        return self.synthesizer.estimateCenter(<int *>labels.data, &vertmap[0, 0, 0], &extents[0, 0], \
                                               height, width, num_classes, preemptive_batch, fx, fy, px, py, &centers[0, 0])
