/*
Copyright (c) 2016, TU Dresden
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "types.h"

#include <string>
#include <vector>
 
/** Here global parameters are defined which are made available throughout the code. */
 
/**
 * @brief Parameters that affect the pose estimation, mainly RANSAC.
 */
struct TestingParameters
{
    bool displayWhileTesting; // stops after each frame to display results (batch mode if false)
    bool rotationObject; // is the object rotation symmetric (affects the calculation of Hinterstoisser pose error)
    
    int testObject; // which test image to load?
    int searchObject; // which object to look for?
    
    int ransacIterations; // initial number of pose hypotheses drawn per frame
    int ransacMaxDraws; // number of times it is tried to draw a valid hypothesis (hypotheses can be rejected if certain contraints are violated)
    int ransacRefinementIterations; // number of iterations for final pose refinement with the general purpose optimizer (refinement using uncertainty)
    int ransacBatchSize; // how many pixels are additionally checked in each iteration of preemptive RANSAC?
    int ransacMaxInliers; // maximal number of inlier correspondences that are used to recalculate poses in coarse refinement
    int ransacMinInliers; // necessary number of inlier correspondences before attempting refinement using uncertainty
    bool ransacRefine; // do refinement using uncertainty?
    int ransacCoarseRefinementIterations; // minimal number of times each final hypothesis must be coarsely refined (recalculating the pose based on inlier correspondences)
    float ransacInlierThreshold2D; // inlier threshold in the RGB case (in px)
    float ransacInlierThreshold3D; // inlier threshold in the RGB-D case (in mm), also used to evaluate the quality of the intermediate object coordinate output
    
    int imageSubSample; // only look at every n th test image (skipping all others), for some quick testing
};

/**
 * @brief Parameters that affect training the random forest and the data.
 */
struct ForestParameters
{
    unsigned treeCount; // how many trees to train for each auto-context layer
    unsigned maxDepth; // tree branches are only grown until this depth

    unsigned acPasses; // number of auto-context layers
    unsigned acSubsample; // auto-context feature channels can be kept sub-sampled in order to save memory
    
    unsigned featureCount; // how many features to try out per node during forest training
    unsigned maxOffset; // maximal offset of the forest features (in px for RGB or in px*m for RGB-D)
    
    unsigned fBGRWeight; // weight of sampling RGB difference features during forest training
    unsigned fACCWeight; // weight of sampling auto-context object class features during forest training (only used after auto-context layer one)
    unsigned fACRWeight; // weight of sampling auto-context object coordinate features during forest training (only used after auto-context layer one)
    
    int maxLeafPoints; // maximum number of samples used in each leaf to fit object coordinate distributions
    unsigned minSamples; // a node is not split further during forest training if less samples arrive
    
    int trainingPixelsPerObject; // how many samples to draw per object (in total, will be split among available training images)
    float trainingPixelFactorRegression; // more samples (by this factor) will be drawn when fitting leaf distributions
    float trainingPixelFactorBG; // more samples (by this factor) will be drawn for the background object/class
    
    std::string sessionString; // custom name that can be used to identify a certain forest (will be appended on the file name)
    std::string config; // name of the config file to read (a file that lists parameter values)
    
    float scaleMin; // minimal scale factor when simulating different image scales (data augmentation, RGB case only)
    float scaleMax; // maximal scale factor when simulating different image scales (data augmentation, RGB case only)
    bool scaleRel; // if true, the image scale is calculated relative to the distance in the training image, i.e. the distance in the training image is normalized to be 1m
    
    float meanShiftBandWidth; // kernel size for mean shift when fitting leaf distributions, in mm
    
    //dataset parameters 
    float focalLength; // focal length of the camera
    float xShift; // x position of the principal point
    float yShift; // y position of the principal point
    
    // in case RGB and depth channels are not aligned (e.g. 7 Scenes dataset), a coarse mapping can be applied by rescaling and shifting the RGB channel
    // note that this is just a coarse registration but worked well in our experiments
    // we later implemented an exact registration (by estimating the relative sensor transformation and projecting RGB onto depth), but results where similar
    bool rawData; // true if RGB and depth channels are not registered
    float secondaryFocalLength; // focal length of the RGB camera 
    float rawXShift; // x shift to be applied to the RGB channel (in px)
    float rawYShift; // y shift to be applied to the RGB channel (in px)
    
    bool fullScreenObject; // true if the objects to estimate fills the entire image, i.e. full scenes (used for calculating the 2D bounding box during optimization)
    
    int imageWidth; // width of the input images (px)
    int imageHeight; // height of the input images (px)
    
    jp::id_t objectCount; // number of objects in the dataset, this is normaly extracted from the folder structure
    jp::cell_t cellSplit; // how often are object coordinates quantized in each dimension (pow(cellsplit, 3) is the number of pseudo classes during forest training)

    /**
     * @brief Returns the number of pseudo classes for each object during forest training.
     * 
     * @return jp::cell_t Number of pseudo classes per object.
     */
    jp::cell_t getCellCount() const { return cellSplit * cellSplit * cellSplit; }
    
    /**
     * @brief Returns the total number of pseudo classes including background.
     * 
     * @return jp::cell_t Total number of pseudo classes.
     */    
    jp::label_t getLabelCount() const { return objectCount * getCellCount() + 1; }

    int maxImageCount; // maximal number of training images to be loaded per object (randomly chosen, have to fit into main memory)
    
    int angleMin; // minimal inplane rotation angle (deg) when simulation image rotations (data augmentation)
    int angleMax; // maximal inplane rotation angle (deg) when simulation image rotations (data augmentation)
    
    bool training; // code runs in training mode? e.g. for applying noise to feature responses during training
    bool useDepth; //RGB or RGBD case?
};

/**
 * @brief Singelton class for providing parameter setting globally throughout the code.
 */
class GlobalProperties
{
protected:
  /**
   * @brief Consgtructor. Sets default values for all parameters.
   */
  GlobalProperties();
public:
    // Forest parameters
    ForestParameters fP;
  
    // Testing parameters
    TestingParameters tP;
    
    /**
     * @brief Get a pointer to the singleton. It will create an instance if none exists yet.
     * 
     * @return GlobalProperties* Singleton pointer.
     */
    static GlobalProperties* getInstance();
    
    /**
     * @brief Returns the file name of the random forest according to the current parameters.
     * 
     * The filename includes some of the most important forest parameters so that changing 
     * these values will not overwrite a previous forest. If you train a forest with a change 
     * in parameters that is not reflected in the file name, use the session string parameter 
     * to differenciate them. 
     * 
     * @param pass Current auto-context layer.
     * @return std::string Random forest file name.
     */
    std::string getFileName(int pass);
    
    /**
     * @brief Returns the 3x3 camera matrix consisting of the intrinsic camera parameters.
     * 
     * @return cv::Mat_< float > Camera/calibration matrix.
     */
    cv::Mat_<float> getCamMat();
    
    std::vector<cv::Mat_<double>> rotations; // a list of pre-calculated image rotation matrices that can be used for simulating image rotations (data augmentation)

    /**
     * @brief Fills the "rotations" member with a number of image rotation matrices. Can be used for data augmentation.
     * 
     * @return void
     */
    void preCalculateRotations();
    
    /**
     * @brief Calculated the FOV from the focal length parameter.
     * 
     * @return float Field of view.
     */
    float getFOV();

    /**
     * @brief Parse the arguments given in the command line and set parameters accordingly
     * 
     * @param argc Number of parameters.
     * @param argv Array of parameters.
     * @return void
     */
    void parseCmdLine(int argc, const char* argv[]);
    
    /**
     * @brief Parse a config file (given by the global parameter "config") and set parameters accordingly
     * 
     * @return void
     */
    void parseConfig();
    
    /**
     * @brief Process a list of arguments and set parameters accordingly.
     * 
     * Each parameter consists of a dash followed by a abbreviation of the parameter. In most cases the next entry in the list should be the value that the parameter should take.
     * 
     * @param argv List of parameters and parameter values. 
     * @return bool
     */
    bool readArguments(std::vector<std::string> argv);
    
private:
    static GlobalProperties* instance; // Singleton instance.
};
