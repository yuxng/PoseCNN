# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import tensorflow as tf
import subprocess

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

def includes_from_flags(flags):
    return [f[2:] for f in flags if f.startswith('-I')]

def lib_libdir_from_flags(flags):
    return [f[2:] for f in flags if f.startswith('-L')], [f[2:] for f in flags if f.startswith('-l')]

tf_compile_flags = [f for f in tf.sysconfig.get_compile_flags() if not f.startswith('-I')]
tf_include_dirs = includes_from_flags(tf.sysconfig.get_compile_flags())

opencv_cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv-3.3.1-dev']).split()
opencv_includes = includes_from_flags(opencv_cflags)
opencv_libs = subprocess.check_output(['pkg-config', '--libs', 'opencv-3.3.1-dev']).split()
opencv_libdirs, opencv_libs = lib_libdir_from_flags(opencv_libs)

def custom_tf_op(name, sources, use_opencv=False):
    ext = Extension(name, sources,
        library_dirs=[CUDA['lib64'], tf.sysconfig.get_lib()],
        libraries=['cudart', 'tensorflow_framework'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={'gcc': ['-std=c++11',
                                    '-D GOOGLE_CUDA=1']+tf_compile_flags,
                            'nvcc': ['-std=c++11',
                                     '-D GOOGLE_CUDA=1',
                                     '-D_MWAITXINTRIN_H_INCLUDED']
                            +tf_compile_flags+
                                     ['-Xcompiler',
                                      '-fPIC',
                                      '-arch=sm_50']},
        include_dirs = tf_include_dirs+[CUDA['include'], '/usr/include/eigen3']
        )
    if use_opencv:
        ext.include_dirs += opencv_includes
        ext.libraries += opencv_libs
        ext.library_dirs += opencv_libdirs
    return ext

ext_modules = [
    custom_tf_op('average_distance_loss.average_distance_loss',
        ['average_distance_loss/average_distance_loss_op_gpu.cu',
         'average_distance_loss/average_distance_loss_op.cc']),
    custom_tf_op('hough_voting_gpu_layer.hough_voting_gpu',
        ['hough_voting_gpu_layer/hough_voting_gpu_op_cc.cu',
         'hough_voting_gpu_layer/hough_voting_gpu_op.cc'],
        use_opencv=True),
    custom_tf_op('hough_voting_layer.houg_voting',
        ['hough_voting_layer/Hypothesis.cpp',
         'hough_voting_layer/thread_rand.cpp',
         'hough_voting_layer/hough_voting_op.cc'],
        use_opencv = True),
    custom_tf_op('roi_pooling_layer.roi_pooling_layer',
        ['roi_pooling_layer/roi_pooling_op_gpu.cu',
         'roi_pooling_layer/roi_pooling_op.cc']),
    custom_tf_op('triplet_loss.triplet_loss',
        ['triplet_loss/triplet_loss_op_gpu.cu',
         'triplet_loss/triplet_loss_op.cc']),
    custom_tf_op('lifted_structured_loss.lifted_structured_loss',
        ['lifted_structured_loss/lifted_structured_loss_op_gpu.cu',
         'lifted_structured_loss/lifted_structured_loss_op.cc']),
    custom_tf_op('computing_flow_layer.computing_flow_layer',
        ['computing_flow_layer/computing_flow_op_gpu.cu',
         'computing_flow_layer/computing_flow_op.cc']),
    custom_tf_op('backprojecting_layer.backprojecting',
        ['backprojecting_layer/backprojecting_op_gpu.cu',
         'backprojecting_layer/backprojecting_op.cc']),
    custom_tf_op('projecting_layer.projecting',
        ['projecting_layer/projecting_op_gpu.cu',
         'projecting_layer/projecting_op.cc']),
    custom_tf_op('computing_label_layer.computing_label',
        ['computing_label_layer/computing_label_op_gpu.cu',
         'computing_label_layer/computing_label_op.cc']),
    custom_tf_op('gradient_reversal_layer.gradient_reversal',
        ['gradient_reversal_layer/gradient_reversal_op_gpu.cu',
         'gradient_reversal_layer/gradient_reversal_op.cc']),
    Extension('normals.gpu_normals',
        ['normals/compute_normals.cu', 'normals/gpu_normals.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler() below
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_35',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include'], '/usr/include/eigen3']
    ),
    Extension(
        "utils.cython_bbox",
        ["utils/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension('nms.gpu_nms',
        ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler() below
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_52',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include']]
    ),
    Extension(
        "synthesize.synthesizer",                                # the extension name
        sources=['synthesize/synthesizer.pyx'],
        language='c++',
        extra_objects=["synthesize/build/libsynthesizer.so"],
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_35',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]}
    )
    #Extension(
    #    "kinect_fusion.kfusion",                                # the extension name
    #    sources=['kinect_fusion/kfusion.pyx'],
    #    language='c++',
    #    extra_objects=["kinect_fusion/build/libkfusion.so"],
    #    extra_compile_args={'gcc': ["-Wno-unused-function"],
    #                        'nvcc': ['-arch=sm_35',
    #                                 '--ptxas-options=-v',
    #                                 '-c',
    #                                 '--compiler-options',
    #                                 "'-fPIC'"]},
    #    include_dirs = ['/usr/local/include/eigen3', '/usr/local/cuda/include', 'kinect_fusion/include']
    #)
]

setup(
    name='fcn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)
