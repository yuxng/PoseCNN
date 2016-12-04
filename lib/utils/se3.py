# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np

# RT is a 3x4 matrix
def se3_inverse(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3,1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3,1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3,1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new
    

