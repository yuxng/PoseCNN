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
3. Download the VGG16 weights from [here](https://drive.google.com/open?id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL) (528M). Put the weight file vgg16.npy to $ROOT/data/imagenet_models.

4. Compile lib/synthesize with cmake (optional). This package contains a few useful tools such as generating synthetic image and ICP.

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

    Compile the Cython interface for lib/synthesize
    ```Shell
    cd $ROOT/lib
    python setup.py build_ext --inplace
    ```

    Add the libary path
    ```Shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT/lib/synthesize/build
    ```

### Tested environment
- Ubuntu 16.04
- Tensorflow >= 1.2.0
- CUDA >= 8.0

### Running the demo
1. Download our trained model on the YCB-Video dataset from [here](https://drive.google.com/open?id=1Zv1cRhFViUkrRi_srqMg5u5Tac8FU3EU), and save it to $ROOT/data/demo_models.

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
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # training
    ./experiments/scripts/lov_color_2d_train.sh $GPU_ID

    # testing
    ./experiments/scripts/lov_color_2d_test.sh $GPU_ID

    ```
