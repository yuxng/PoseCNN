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

#include "properties.h"
#include "util.h"
#include "thread_rand.h"

#include <iostream>
#include <fstream>
#include <valarray>

GlobalProperties* GlobalProperties::instance = NULL;

GlobalProperties::GlobalProperties()
{
    // forest parameters
    fP.treeCount = 3;
    fP.maxDepth = 64;

    fP.acPasses = 1;
    fP.acSubsample = 1;

    fP.featureCount = 1000;
    fP.maxOffset = 20;

    fP.fBGRWeight = 1;
    fP.fACCWeight = 1;
    fP.fACRWeight = 1;
    
    fP.maxLeafPoints = 2000;
    fP.minSamples = 50;
    
    fP.trainingPixelsPerObject = 500000;
    fP.trainingPixelFactorRegression = 5;
    fP.trainingPixelFactorBG = 3;
    
    fP.sessionString = "";
    fP.config = "default";
    
    fP.scaleMin = 0.5;
    fP.scaleMax = 2;
    fP.scaleRel = false;
    
    fP.meanShiftBandWidth = 100.0;
    
    //dataset parameters 
    fP.focalLength = 585.0f;
    fP.xShift = 0.f;
    fP.yShift = 0.f;
    
    fP.rawData = false;
    fP.secondaryFocalLength = 525.0f;
    fP.rawXShift = 0;
    fP.rawYShift = 0;
    
    fP.fullScreenObject = false;
    
    fP.imageWidth = 640;
    fP.imageHeight = 480;
    
    fP.objectCount = 0;
    fP.cellSplit = 5;
    fP.maxImageCount = -1;
    
    fP.angleMax = 0;
    fP.angleMin = 0;
    
    fP.useDepth = false;
    
    fP.training = false;
    
    // testing parameters
    tP.displayWhileTesting = true;
    tP.rotationObject = false;

    tP.ransacIterations = 256;
    tP.ransacMaxDraws = 10000000;
    tP.ransacCoarseRefinementIterations = 8;
    tP.ransacRefinementIterations = 100;
    tP.ransacBatchSize = 1000;
    tP.ransacMaxInliers = 1000;
    tP.ransacMinInliers = 10;
    tP.ransacRefine = true;
    tP.ransacInlierThreshold2D = 10;
    tP.ransacInlierThreshold3D = 100;

    tP.imageSubSample = 1;
    
    tP.testObject = 1;
    tP.searchObject = -1;
}

GlobalProperties* GlobalProperties::getInstance()
{
    if(instance == NULL)
    instance = new GlobalProperties();
    return instance;
}
  
std::string GlobalProperties::getFileName(int pass)
{
    std::string baseFileName = 
	"pass" + intToString(pass) +
	"_fc" + intToString(fP.featureCount) +
	"_md" + intToString(fP.maxDepth) + 
	"_mo" + intToString(fP.maxOffset) +
	"_wbgr" + intToString((int)(fP.fBGRWeight)) +
	"_wacc" + intToString((int)(fP.fACCWeight)) +
	"_wacr" + intToString((int)(fP.fACRWeight)) +
	"_mlp" + intToString(fP.maxLeafPoints) +
	"_tp" + intToString(fP.trainingPixelsPerObject) +
	"_tfr" + intToString(fP.trainingPixelFactorRegression) +
	"_tfb" + intToString(fP.trainingPixelFactorBG) +
	"_ms" + intToString(fP.minSamples) +
	"_cs" + intToString(fP.cellSplit) +
	"_tc" + intToString(fP.treeCount) +
	"_" + fP.sessionString + 
	".rf";

    return baseFileName;
}
    
bool GlobalProperties::readArguments(std::vector<std::string> argv)
{
    int argc = argv.size();

    for(int i = 0; i < argc; i++)
    {
	std::string s = argv[i];
	
	if(s == "-fc")
	{ 
	    i++;
	    fP.featureCount = std::atoi(argv[i].c_str());
	    std::cout << "feature count: " << fP.featureCount << "\n";   
	    continue;
	}

	if(s == "-md")
	{ 
	    i++;
	    fP.maxDepth = std::atoi(argv[i].c_str());
	    std::cout << "maximum depth: " << fP.maxDepth << "\n";   
	    continue;
	}

	if(s == "-iw")
	{ 
	    i++;
	    fP.imageWidth = std::atoi(argv[i].c_str());
	    std::cout << "image width: " << fP.imageWidth << "\n";   
	    continue;
	}
	
	if(s == "-ih")
	{ 
	    i++;
	    fP.imageHeight = std::atoi(argv[i].c_str());
	    std::cout << "image height: " << fP.imageHeight << "\n";   
	    continue;
	}	
	
	if(s == "-fl")
	{ 
	    i++;
	    fP.focalLength = (float)std::atof(argv[i].c_str());
	    std::cout << "focal length: " << fP.focalLength << "\n";   
	    continue;
	}
	
	if(s == "-xs")
	{ 
	    i++;
	    fP.xShift = (float)std::atof(argv[i].c_str());
	    std::cout << "x shift: " << fP.xShift << "\n";   
	    continue;
	}
	
	if(s == "-ys")
	{ 
	    i++;
	    fP.yShift = (float)std::atof(argv[i].c_str());
	    std::cout << "y shift: " << fP.yShift << "\n";   
	    continue;
	}	
	
	if(s == "-rd")
	{ 
	    i++;
	    fP.rawData = std::atoi(argv[i].c_str());
	    std::cout << "raw data (rescale rgb): " << fP.rawData << "\n";   
	    continue;
	}
	
	if(s == "-fso")
	{ 
	    i++;
	    fP.fullScreenObject = std::atoi(argv[i].c_str());
	    std::cout << "full screen object: " << fP.fullScreenObject << "\n";   
	    continue;
	}	
	
	if(s == "-sfl")
	{ 
	    i++;
	    fP.secondaryFocalLength = (float)std::atof(argv[i].c_str());
	    std::cout << "secondary focal length: " << fP.secondaryFocalLength << "\n";   
	    continue;
	}	
	
	if(s == "-rxs")
	{ 
	    i++;
	    fP.rawXShift = (float)std::atof(argv[i].c_str());
	    std::cout << "raw x shift: " << fP.rawXShift << "\n";   
	    continue;
	}	
	
	if(s == "-rys")
	{ 
	    i++;
	    fP.rawYShift = (float)std::atof(argv[i].c_str());
	    std::cout << "raw y shift: " << fP.rawYShift << "\n";   
	    continue;
	}		
	
	if(s == "-wbgr")
	{ 
	    i++;
	    fP.fBGRWeight = (float)std::atof(argv[i].c_str());
	    std::cout << "bgr feature weight: " << fP.fBGRWeight << "\n";   
	    continue;
	}

	if(s == "-wacc")
	{ 
	    i++;
	    fP.fACCWeight = (float)std::atof(argv[i].c_str());
	    std::cout << "auto-context class feature weight: " << fP.fACCWeight << "\n";   
	    continue;
	}

	if(s == "-wacr")
	{ 
	    i++;
	    fP.fACRWeight = (float)std::atof(argv[i].c_str());
	    std::cout << "auto-context regression feature weight: " << fP.fACRWeight << "\n";   
	    continue;
	}

	if(s == "-smin")
	{ 
	    i++;
	    fP.scaleMin = (float)std::atof(argv[i].c_str());
	    std::cout << "min scale: " << fP.scaleMin << "\n";   
	    continue;
	}	

	if(s == "-smax")
	{ 
	    i++;
	    fP.scaleMax = (float)std::atof(argv[i].c_str());
	    std::cout << "max scale: " << fP.scaleMax << "\n";   
	    continue;
	}	
	
	if(s == "-srel")
	{ 
	    i++;
	    fP.scaleRel = std::atoi(argv[i].c_str());
	    std::cout << "relative scale: " << fP.scaleRel << "\n";   
	    continue;
	}
	
	if(s == "-ud")
	{ 
	    i++;
	    fP.useDepth = std::atoi(argv[i].c_str());
	    std::cout << "use depth: " << fP.useDepth << "\n";   
	    continue;
	}	
	
	if(s == "-rR")
	{ 
	    i++;
	    tP.ransacRefine = std::atoi(argv[i].c_str());
	    std::cout << "ransac refine: " << tP.ransacRefine << "\n";   
	    continue;
	}		

	if(s == "-mi")
	{ 
	    i++;
	    fP.maxImageCount = std::atoi(argv[i].c_str());
	    std::cout << "max training images: " << fP.maxImageCount << "\n";   
	    continue;
	}	
	
	if(s == "-tc")
	{ 
	    i++;
	    fP.treeCount = std::atoi(argv[i].c_str());
	    std::cout << "tree count: " << fP.treeCount << "\n";   
	    continue;
	}

	if(s == "-mo")
	{ 
	    i++;
	    fP.maxOffset = std::atoi(argv[i].c_str());
	    std::cout << "maximum feature offset: " << fP.maxOffset << "\n";   
	    continue;
	}

	if(s == "-mlp")
	{ 
	    i++;
	    fP.maxLeafPoints = std::atoi(argv[i].c_str());
	    std::cout << "maximum leaf points: " << fP.maxLeafPoints << "\n";   
	    continue;
	}

	if(s == "-ms")
	{ 
	    i++;
	    fP.minSamples = std::atoi(argv[i].c_str());
	    std::cout << "minimum node samples: " << fP.minSamples << "\n";   
	    continue;
	}
	
	if(s == "-mb")
	{ 
	    i++;
	    fP.meanShiftBandWidth = std::atof(argv[i].c_str());
	    std::cout << "mean-shift band width: " << fP.meanShiftBandWidth << "\n";   
	    continue;
	}	
	    
	if(s == "-tp")
	{ 
	    i++;
	    fP.trainingPixelsPerObject = std::atoi(argv[i].c_str());
	    std::cout << "training samples per object: " << fP.trainingPixelsPerObject << "\n";   
	    continue;
	}

	if(s == "-tfr")
	{ 
	    i++;
	    fP.trainingPixelFactorRegression = std::atof(argv[i].c_str());
	    std::cout << "samples factor for regression: " << fP.trainingPixelFactorRegression << "\n";   
	    continue;
	}

	if(s == "-tfb")
	{ 
	    i++;
	    fP.trainingPixelFactorBG = std::atof(argv[i].c_str());
	    std::cout << "samples factor for background: " << fP.trainingPixelFactorBG << "\n";   
	    continue;
	}	

	if(s == "-amin")
	{ 
	    i++;
	    fP.angleMin = std::atoi(argv[i].c_str());
	    std::cout << "in-plane rotation minimum: " << fP.angleMin << "\n";   
	    preCalculateRotations();
	    continue;
	}

	if (s == "-amax")
	{ 
	    i++;
	    fP.angleMax = std::atoi(argv[i].c_str());
	    std::cout << "in-plane rotation maximum: " << fP.angleMax << "\n";   
	    preCalculateRotations();
	    continue;
	}

	if(s == "-cs")
	{ 
	    i++;
	    fP.cellSplit = std::atoi(argv[i].c_str());
	    std::cout << "cell split: " << fP.cellSplit << "\n";   
	    continue;
	}

	if(s == "-sid")
	{ 
	    i++;
	    fP.sessionString = argv[i];
	    std::cout << "session string: " << fP.sessionString << "\n";
	    fP.config = fP.sessionString;
	    parseConfig();
	    continue;
	}

	if(s == "-acp")
	{ 
	    i++;
	    fP.acPasses = std::atoi(argv[i].c_str());
	    std::cout << "auto-context passes: " << fP.acPasses << "\n";   
	    continue;
	}
	
	if(s == "-acs")
	{ 
	    i++;
	    fP.acSubsample = std::atoi(argv[i].c_str());
	    std::cout << "auto-context sub-sampling: " << fP.acSubsample << "\n";   
	    continue;
	}	
	
	// test parameters
	if(s == "-nD")
	{
	    tP.displayWhileTesting = false;
	    std::cout << "display test output: " << tP.displayWhileTesting << "\n";   
	    continue;
	}

	if(s == "-rO")
	{
	    tP.rotationObject = true;
	    std::cout << "rotation-symmetric object: " << tP.rotationObject << "\n";   
	    continue;
	}	   

	if(s == "-tO")
	{
	    i++;
	    tP.testObject = std::atoi(argv[i].c_str());
	    std::cout << "test object: " << tP.testObject << "\n";   
	    continue;
	}
	
	if(s == "-sO")
	{
	    i++;
	    tP.searchObject = std::atoi(argv[i].c_str());
	    std::cout << "search object: " << tP.searchObject << "\n";   
	    continue;
	}	

	if(s == "-rI")
	{
	    i++;
	    tP.ransacIterations = std::atoi(argv[i].c_str());
	    std::cout << "ransac iterations: " << tP.ransacIterations << "\n";   
	    continue;
	}
	
	if(s == "-rB")
	{
	    i++;
	    tP.ransacBatchSize = std::atoi(argv[i].c_str());
	    std::cout << "ransac batch size: " << tP.ransacBatchSize << "\n";   
	    continue;
	}	

	if(s == "-rT2D")
	{
	    i++;
	    tP.ransacInlierThreshold2D = (float)std::atof(argv[i].c_str());
	    std::cout << "ransac inlier threshold: " << tP.ransacInlierThreshold2D << "\n";   
	    continue;
	}

	if(s == "-rT3D")
	{
	    i++;
	    tP.ransacInlierThreshold3D = (float)std::atof(argv[i].c_str());
	    std::cout << "ransac inlier threshold: " << tP.ransacInlierThreshold3D << "\n";   
	    continue;
	}	
	
	if(s == "-rRI")
	{
	    i++;
	    tP.ransacRefinementIterations = std::atoi(argv[i].c_str());
	    std::cout << "ransac iterations (refinement): " << tP.ransacRefinementIterations << "\n";   
	    continue;
	}
	
	if(s == "-rMD")
	{
	    i++;
	    tP.ransacMaxDraws = std::atoi(argv[i].c_str());
	    std::cout << "ransac maximum hypotheses drawn: " << tP.ransacMaxDraws << "\n";   
	    continue;
	}	
	
	if(s == "-rCRI")
	{
	    i++;
	    tP.ransacCoarseRefinementIterations = std::atoi(argv[i].c_str());
	    std::cout << "ransac iterations (coarse refinement): " << tP.ransacCoarseRefinementIterations << "\n";   
	    continue;
	}	
	
	if(s == "-rMI")
	{
	    i++;
	    tP.ransacMaxInliers = std::atoi(argv[i].c_str());
	    std::cout << "ransac maximum inliers: " << tP.ransacMaxInliers << "\n";   
	    continue;
	}	
	
	if(s == "-rMinI")
	{
	    i++;
	    tP.ransacMinInliers = std::atoi(argv[i].c_str());
	    std::cout << "ransac minimum inliers: " << tP.ransacMinInliers << "\n";   
	    continue;
	}		

	if(s == "-iSS")
	{
	    i++;
	    tP.imageSubSample = std::atoi(argv[i].c_str());
	    std::cout << "test image sub sampling: " << tP.imageSubSample << "\n";   
	    continue;
	}
	
	std::cout << "unkown argument: " << argv[i] << "\n";
	return false;
    }
}
  
void GlobalProperties::parseCmdLine(int argc, const char* argv[])
{
    std::vector<std::string> argVec;
    for(int i = 1; i < argc; i++) argVec.push_back(argv[i]);
    readArguments(argVec);
}

void GlobalProperties::parseConfig()
{
    std::string configFile = fP.config + ".config";
    std::cout << BLUETEXT("Parsing config file: ") << configFile << std::endl;
    
    std::ifstream file(configFile);
    if(!file.is_open()) return;

    std::vector<std::string> argVec;

    std::string line;
    std::vector<std::string> tokens;
	
    while(true)
    {
	if(file.eof()) break;
	
	std::getline(file, line);
	if(line.length() == 0) continue; //empty line
	if(line.at(0) == '#') continue; // comment
	
	tokens = split(line);
	if(tokens.empty()) continue;
	
	argVec.push_back("-" + tokens[0]);
	argVec.push_back(tokens[1]);
    }
    
    readArguments(argVec);
}

void GlobalProperties::preCalculateRotations()
{
    std::cout << "Precalculating Rotations" << std::endl;
    rotations.clear();
    
    for(unsigned r = 0; r < 1000; r++)
	rotations.push_back(cv::getRotationMatrix2D(cv::Point2f(0, 0), drand(fP.angleMin, fP.angleMax), 1.0));
}

cv::Mat_<float> GlobalProperties::getCamMat()
{
    float centerX = fP.imageWidth / 2 + fP.xShift;
    float centerY = fP.imageHeight / 2 + fP.yShift;
    float f = fP.focalLength; 

    cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
    camMat(0, 0) = f;
    camMat(1, 1) = f;
    camMat(2, 2) = 1.f;
    
    camMat(0, 2) = centerX;
    camMat(1, 2) = centerY;
    
    return camMat;
}

float GlobalProperties::getFOV()
{
    return 2 * atan((fP.imageHeight/2.0)/fP.focalLength)*180/CV_PI;
}

static GlobalProperties* instance;