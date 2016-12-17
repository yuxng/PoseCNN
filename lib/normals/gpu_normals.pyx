# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_normals.hpp":
    void compute_normals(np.float32_t*, np.float32_t*, float, float, float, float, float, int, int, int)

def gpu_normals(np.ndarray[np.float32_t, ndim=2] depth, np.float32_t fx, np.float32_t fy, np.float32_t cx, np.float32_t cy, np.float32_t depthCutoff, np.int32_t device_id=0):
    cdef int height = depth.shape[0]
    cdef int width = depth.shape[1]

    cdef np.ndarray[np.float32_t, ndim=3] \
        nmap = np.zeros((height, width, 3), dtype=np.float32)

    compute_normals(&depth[0, 0], &nmap[0, 0, 0], fx, fy, cx, cy, depthCutoff, height, width, device_id)
    return nmap
