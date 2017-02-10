# DA-RNN: Semantic Mapping with Data Associated Recurrent Neural Networks

Created by Yu Xiang at RSE-Lab at University of Washington.

### Introduction

we introduce Data Associated Recurrent Neural Networks (DA-RNNs), a novel framework for joint 3D scene mapping and semantic labeling. DA-RNNs use a new recurrent neural network architecture for semantic labeling on RGB-D videos. The output of the network is integrated with mapping techniques such as KinectFusion in order to inject semantic information into the reconstructed 3D scene.

### Installation

DA-RNN consists a reccurent neural network for semantic labeling on RGB-D videos and the KinectFusion module for 3D reconstruction. The RNN and KinectFusion communicate via a Python interface.

- Install [TensorFlow](https://www.tensorflow.org/get_started/os_setup). I suggest to use the Virtualenv installation.

- Compile the new layers under $ROOT/lib we introduce in DA-RNN.
    ```Shell
    cd $ROOT/lib
    sh make.sh
    ```

- Compile KinectFusion with cmake
    ```Shell
    cd $ROOT/lib/kinect_fusion
    mkdir build
    cd build
    cmake ..
    make
    ```

### License

DA-RNN is released under the MIT License (refer to the LICENSE file for details).
