# PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes

Created by Yu Xiang at [RSE-Lab](http://rse-lab.cs.washington.edu/) at University of Washington.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find PoseCNN useful in your research, please consider citing:

    @inproceedings{xiang2017posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        Journal   = {arXiv preprint arXiv:1711.00199},
        Year = {2017}
    }

### Installation

1. Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup). I usually compile the source code of tensorflow locally.

2. Compile the new layers under $ROOT/lib we introduce in PoseCNN.
    ```Shell
    cd $ROOT/lib
    sh make.sh
    ```

3. Compile lib/synthesize with cmake (optional). This package contains a few useful tools such as generating synthetic image and ICP.

   Install dependencies:
   - [Pangolin](https://github.com/stevenlovegrove/Pangolin)
   - [Eigen](https://eigen.tuxfamily.org)
   - [Sophus](https://github.com/strasdat/Sophus)
   - [nanoflann](https://github.com/jlblancoc/nanoflann)
   - libsuitesparse-dev

    ```Shell
    cd $ROOT/lib/synthesize
    mkdir build
    cd build
    cmake ..
    make
    ```

4. Compile the Cython interface for lib/synthesize
    ```Shell
    cd $ROOT/lib
    python setup.py build_ext --inplace
    ```

5. Add the libary path
    ```Shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT/lib/synthesize/build
    ```

6. Download the VGG16 weights from [here](https://drive.google.com/open?id=0B4WdmTHU8V7VMTducllWZzA0REU) (57M). Put the weight file vgg16_convs.npy to $ROOT/data/imagenet_models.

### Tested environment
- Ubuntu 16.04
- Tensorflow 1.2.0
- CUDA 8.0

### Running on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset
    ```Shell
    cd $ROOT/data/LOV
    ln -s $ycb_data data
    ```

3. Training and testing on the RGB-D Scene dataset
    ```Shell
    cd $ROOT

    # training
    ./experiments/scripts/lov_color_2d_train.sh $GPU_ID

    # testing
    ./experiments/scripts/lov_color_2d_test.sh $GPU_ID

    ```

### Using Our Trained Models
1. You can download all our trained tensorflow models on the RGB-D Scene dataset and the ShapeNet Scene dataset from [here](https://drive.google.com/file/d/0B4WdmTHU8V7VQWFnRmFIVTA1LXc/view?usp=sharing) (3.1G).

    ```Shell
    # an exmaple to test the trained model
    ./experiments/scripts/rgbd_scene_multi_rgbd_test.sh $GPU_ID

    ```
