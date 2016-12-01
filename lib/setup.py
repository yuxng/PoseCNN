# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
        "kinect_fusion.kfusion",                                # the extension name
        sources=['kinect_fusion/kfusion.pyx'],
        language='c++',
        extra_objects=["kinect_fusion/build/libkfusion.so"],
        include_dirs = ['/usr/local/include/eigen3', '/usr/local/cuda/include', 'kinect_fusion/include']
      )))
