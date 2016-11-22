# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "icp_kernel.hpp":
    void _icp(np.float64_t*, np.float64_t*, np.float64_t*, np.float64_t*, np.float64_t, int, int, int, int)

# ICP
# model: N x 3 matrix
# template: N x 3 matrix
# T: 4 x 4 matrix, initial SE3 transformation
# indist: float, inlier distance
# flag: int, 0, point_to_point, 1, point_to_plane
def icp(np.ndarray[np.float64_t, ndim=2] model, np.ndarray[np.float64_t, ndim=2] template, np.ndarray[np.float64_t, ndim=2] T, np.float64_t indist, np.int32_t flag):

    cdef int num_model = model.shape[0]
    cdef int dim = model.shape[1]
    cdef int num_temp = template.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] \
        Tr = np.zeros((dim+1, dim+1), dtype=np.float64)

    _icp(&Tr[0, 0], &model[0, 0], &template[0, 0], &T[0, 0], indist, num_model, num_temp, dim, flag)

    return Tr
