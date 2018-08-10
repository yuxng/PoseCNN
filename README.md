# PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes

Created by Yu Xiang at [RSE-Lab](http://rse-lab.cs.washington.edu/) at University of Washington and NVIDIA Research.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find PoseCNN useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        Journal   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

### Installation

1. Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup) version r1.8 from source binaries won't work because of ABI incompatibilities.
      1. Make a virtualenv with and add these packages, otherwise tf can not be build by source: pip install mock keras Cython easydict transforms3d(the last two are needed for later to actually run the demo)

2. Download the VGG16 weights from [here](https://drive.google.com/open?id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL) (528M). Put the weight file vgg16.npy to $ROOT/data/imagenet_models.

3. Get dependencies: sudo apt-get install libsuitesparse-dev openexr libopenexr-dev metis libmetis-dev

4. Compile lib/kinect_fusion to be able to compile lib/synthesize. 
    1. Add nanoflann to the include dirs in cmake and sohpus to the include and link dirs
    2. Downgrade cmake to version 3.6.0
    3. Install Eigen 3.3.90
    4. If Pangolin is already installed, reinstall Pangolin, since it will be pointing at the old eigen which has the cuda bug mentioned here: https://devtalk.nvidia.com/default/topic/1026622/cuda-programming-and-performance/nvcc-can-t-compile-code-that-uses-eigen/
    5. Upgrade boost to 1.67.0 and move lib to /usr/lib/x86_64-linux-gnu and include boost the whole folder to /usr/include
    

5. Compile lib/synthesize with cmake (optional). This package contains a few useful tools such as generating synthetic images for training and ICP.

   Install dependencies:
   - Python version 2.7, 3.X won't work
   - [Pangolin](https://github.com/stevenlovegrove/Pangolin) commit 1ec721d59ff6b799b9c24b8817f3b7ad2c929b83 worked for me, original author used c2a6ef524401945b493f14f8b5b8aa76cc7d71a9
   - [Eigen](https://eigen.tuxfamily.org) 3.3.90
   - [boost](https://www.boost.org/) 1.67.0
   - [Sophus](https://github.com/strasdat/Sophus) commit ceb6380a1584b300e687feeeea8799353d48859f
   - [nanoflann](https://github.com/jlblancoc/nanoflann) commit ad7547f4e6beb1cdb3e360912fd2e352ef959465
   - [nlopt](https://github.com/stevengj/nlopt) Important install from this github repo, not using the instructions [here](https://nlopt.readthedocs.io/en/latest/) commit 74e647b667f7c4500cdb4f37653e59c29deb9ee2
   

   We use Boost.Python library to link tensorflow with the c++ code. Make sure you have it in your Boost. The tested Boost version is 1.66.0.

   1. Change hard coded pathes in lib/synthesize/CMakeLists.txt of boost and add sophus paths manually or via find_script. Uncomment line #add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0) to fix potential compatibility issues with tf.
   2. Uncomment ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc from include dirs because of this issue: https://github.com/stevenlovegrove/Pangolin/issues/268
   3. Surpress these warnings so you see errors better.
   ```add_definitions(-Wno-sign-compare)
      add_definitions(-Wno-unused-result)
      add_definitions(-Wno-unused-but-set-variable)
      add_definitions(-Wno-unused-variable)
      add_definitions(-Wno-reorder)
      add_definitions(-Wno-deprecated-declarations)
   ```
   4. Adapt boost_python and boost_numpy in Cmake line 98/99 to your library name when using boost and python 3.5 it is boost_python27 and boost_numpy27 or symlink these to boots_python and boost_numpy.
   5. Create folder data and models under data/LOV and add or symlink the data and models into there
   
### Building
   1. Build kinect_fusion
   
    ```Shell
    cd $ROOT/lib/kinect_fusion
    mkdir build
    cd build
    cmake ..
    make
    ```
   2. Build synthesize
   
    ```Shell
    cd $ROOT/lib/synthesize
    mkdir build
    cd build
    cmake ..
    make
    ```
   3. Compile the new layers under $ROOT/lib we introduce in PoseCNN.
    ```Shell
    cd $ROOT/lib
    sh make.sh
    ```
   
   4. run python setup ```python setup.py build_ext --inplace```
   
   5. Add pythonpaths

    Add the path of the built libary libsynthesizer.so to python path
    ```Shell
    export PYTHONPATH=$PYTHONPATH:$ROOT/lib:$ROOT/lib/synthesize/build
    ```

### Required environment
- Ubuntu 16.04
- Tensorflow >= 1.2.0
- CUDA >= 8.0

### Running the demo
1. Download our trained model on the YCB-Video dataset from [here](https://drive.google.com/file/d/1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz/view?usp=sharing), and save it to $ROOT/data/demo_models.

2. run the following script
    ```Shell
    ./experiments/scripts/demo.sh $GPU_ID
    ```

### Running on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset (the name LOV is due to legacy, Learning Objects from Videos)
    ```Shell
    cd $ROOT/data/LOV
    ln -s $ycb_data data
    ln -s $ycb_models models
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # training
    ./experiments/scripts/lov_color_2d_train.sh $GPU_ID

    # testing
    ./experiments/scripts/lov_color_2d_test.sh $GPU_ID

    ```
