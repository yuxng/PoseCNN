#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
#include <nlopt.hpp>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

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
#include <OpenEXR/half.h>

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

class Synthesizer
{
 public:

  Synthesizer() {};
  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer();

  void create_window(int width, int height);
  void destroy_window();
  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float* class_indexes, float *poses_return, float* centers_return,
              float* vertex_targets, float* vertex_weights, float weight);
  void loadModels(std::string filename);
  void loadPoses(const std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);

 private:
  int counter_;

  // poses
  std::vector<float*> poses_;
  std::vector<int> pose_nums_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;
  std::vector<std::string> texture_names_;

  // pangoline views
  pangolin::View* gtView_;

  // buffers
  std::vector<pangolin::GlBuffer> texturedVertices_;
  std::vector<pangolin::GlBuffer> canonicalVertices_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<df::CanonicalVertRenderType>* renderer_;
};
