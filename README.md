# DA-RNN: Semantic Mapping with Data Associated Recurrent Neural Networks

Created by Yu Xiang at RSE-Lab at University of Washington.

### Introduction

we introduce Data Associated Recurrent Neural Networks (DA-RNNs), a novel framework for joint 3D scene mapping and semantic labeling. DA-RNNs use a new recurrent neural network architecture for semantic labeling on RGB-D videos. The output of the network is integrated with mapping techniques such as KinectFusion in order to inject semantic information into the reconstructed 3D scene.

### Installation

DA-RNN consists a reccurent neural network for semantic labeling on RGB-D videos and the KinectFusion module for 3D reconstruction. The RNN and KinectFusion communicate via a Python interface.

1. Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup). I suggest to use the Virtualenv installation.

2. Compile the new layers under $ROOT/lib we introduce in DA-RNN.
    ```Shell
    cd $ROOT/lib
    sh make.sh
    ```

3. Compile KinectFusion with cmake. Unfortunately, this step requires some effort.

   Install dependencies of KinectFusion:
   - [Pangolin](https://github.com/stevenlovegrove/Pangolin)
   - [Eigen](https://eigen.tuxfamily.org)
   - [Sophus](https://github.com/strasdat/Sophus)
   - [nanoflann](https://github.com/jlblancoc/nanoflann)

    ```Shell
    cd $ROOT/lib/kinect_fusion
    mkdir build
    cd build
    cmake ..
    make
    ```

4. Compile the Cython interface for RNN and KinectFusion
    ```Shell
    cd $ROOT/lib
    python setup.py build_ext --inplace
    ```

### Running on the RGB-D Scene dataset
1. Download the RGB-D Scene dataset from here.

2. Create a symlink for the RGB-D Scene dataset
    ```Shell
    cd $ROOT/data/RGBDScene
    ln -s $RGB-D_scene_data data
    ```

3. Training and testing on the RGB-D Scene dataset
    ```Shell
    cd $ROOT
    ./experiments/scripts/rgbd_scene_multi_rgbd.sh $GPU_ID

    ```

### License

DA-RNN is released under the MIT License (refer to the LICENSE file for details).
