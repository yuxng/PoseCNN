# DA-RNN: Semantic Mapping with Data Associated Recurrent Neural Networks

Created by Yu Xiang and Tanner Schmidt at [RSE-Lab](http://rse-lab.cs.washington.edu/) at University of Washington.

### Introduction

We introduce Data Associated Recurrent Neural Networks (DA-RNNs), a novel framework for joint 3D scene mapping and semantic labeling. DA-RNNs use a new recurrent neural network architecture for semantic labeling on RGB-D videos. The output of the network is integrated with mapping techniques such as KinectFusion in order to inject semantic information into the reconstructed 3D scene. [arXiv](https://arxiv.org/abs/1703.03098), [Video](https://youtu.be/5vnw7ZrZlB8)

[![DA-RNN](http://yuxng.github.io/DA-RNN.png)](https://youtu.be/5vnw7ZrZlB8)

### License

DA-RNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find DA-RNN useful in your research, please consider citing:

    @inproceedings{xiang2017darnn,
        Author = {Yu Xiang and Dieter Fox},
        Title = {DA-RNN: Semantic Mapping with Data Associated Recurrent Neural Networks},
        Booktitle = {Robotics: Science and Systems (RSS)},
        Year = {2017}
    }

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
   - [Sophus](https://github.com/strasdat/Sophus/tree/v0.9.5)
   - [nanoflann](https://github.com/jlblancoc/nanoflann)
   - libsuitesparse-dev

    ```Shell
    cd $ROOT/lib/kinect_fusion
    mkdir build
    cd build
    cmake ..
    make
    ```

    **Note:** If you see errors in calling Eigen like "calling a host function from a device function", please download and use our modified Eigen from [here](https://drive.google.com/open?id=0B4WdmTHU8V7VTDFIdU5IWGxkMGM).

4. Compile the Cython interface for RNN and KinectFusion
    ```Shell
    cd $ROOT/lib
    python setup.py build_ext --inplace
    ```

5. Add the KinectFusion libary path
    ```Shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT/lib/kinect_fusion/build
    ```

6. Download the VGG16 weights from [here](https://drive.google.com/open?id=0B4WdmTHU8V7VMTducllWZzA0REU) (57M). Put the weight file vgg16_convs.npy to $ROOT/data/imagenet_models.

### Tested environment
- Ubuntu 16.04
- Tensorflow 1.2.0
- CUDA 8.0

### Running on the RGB-D Scene dataset
1. Download the RGB-D Scene dataset from [here](https://drive.google.com/open?id=0B4WdmTHU8V7VaHIxckxwbVpabFU) (5.5G).

2. Create a symlink for the RGB-D Scene dataset
    ```Shell
    cd $ROOT/data/RGBDScene
    ln -s $RGB-D_scene_data data
    ```

3. Training and testing on the RGB-D Scene dataset
    ```Shell
    cd $ROOT

    # train and test RNN with different input (color, depth, normal and rgbd)
    ./experiments/scripts/rgbd_scene_multi_*.sh $GPU_ID

    # train and test FCN with different input (color, depth, normal and rgbd)
    ./experiments/scripts/rgbd_scene_single_*.sh $GPU_ID

    ```

### Running on the ShapeNet Scene dataset
1. Download the ShapeNet Scene dataset from [here](https://drive.google.com/open?id=0B4WdmTHU8V7VTzRfZTFPd0JKYTg) (2.3G).

2. Create a symlink for the ShapeNet Scene dataset
    ```Shell
    cd $ROOT/data/ShapeNetScene
    ln -s $ShapeNet_scene_data data
    ```

3. Training and testing on the RGB-D Scene dataset
    ```Shell
    cd $ROOT

    # train and test RNN with different input (color, depth, normal and rgbd)
    ./experiments/scripts/shapenet_scene_multi_*.sh $GPU_ID

    # train and test FCN with different input (color, depth, normal and rgbd)
    ./experiments/scripts/shapenet_scene_single_*.sh $GPU_ID

    ```

### Using Our Trained Models
1. You can download all our trained tensorflow models on the RGB-D Scene dataset and the ShapeNet Scene dataset from [here](https://drive.google.com/file/d/0B4WdmTHU8V7VQWFnRmFIVTA1LXc/view?usp=sharing) (3.1G).

    ```Shell
    # an exmaple to test the trained model
    ./experiments/scripts/rgbd_scene_multi_rgbd_test.sh $GPU_ID

    ```
