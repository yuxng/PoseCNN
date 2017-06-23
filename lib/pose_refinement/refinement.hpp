#include <stdio.h>
#include <cfloat>
#include <math.h> 
#include <vector>
#include <ctime>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstddef> 

#include <pangolin/pangolin.h>
#include "opencv2/opencv.hpp"

#include <Eigen/Core>

#include <df/camera/camera.h>
#include <df/camera/linear.h>
#include <df/camera/poly3.h>
#include <df/camera/rig.h>
#include <df/image/backprojection.h>
#include <df/prediction/glRender.h>
#include <df/prediction/glRenderTypes.h>
#include <df/util/args.h>
#include <df/util/glHelpers.h>
#include <df/util/pangolinHelpers.h>
#include <df/util/tensor.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>

template <typename Derived>
inline void operator >>(std::istream & stream, Eigen::MatrixBase<Derived> & M)
{

    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            stream >> M(r,c);
        }
    }

}

unsigned char class_colors[22][3] = {{255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                              {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
                              {64, 0, 0}, {0, 64, 0}, {0, 0, 64}, {64, 64, 0}, {64, 0, 64}, {0, 64, 64}, 
                              {192, 0, 0}, {0, 192, 0}, {0, 0, 192}};


class Refiner
{
 public:

  Refiner() {};
  Refiner(std::string model_file);
  ~Refiner();

  void setup(std::string model_file);
  void create_window(int width, int height);
  void destroy_window();
  void render(unsigned char* data, unsigned char* labels, float* rois, int num_rois, int num_gt, int width, int height,
                    float* poses_gt, float* poses_pred, float fx, float fy, float px, float py);
  void loadModels(std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(aiMesh* assimpMesh, std::string textureName, 
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);
  void feed_data(int width, int height, unsigned char* data, unsigned char* labels, pangolin::GlTexture & colorTex, pangolin::GlTexture & labelTex);

 private:
  int counter_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;
  std::vector<std::string> texture_names_;

  // pangoline views
  pangolin::View* gtView_;
  pangolin::View* poseView_;
  pangolin::View* colorView_;
  pangolin::View* labelView_;
  pangolin::View* multiView_;
};
