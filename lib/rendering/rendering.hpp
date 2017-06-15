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

#include <Eigen/Sparse>

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

struct ForegroundRenderType {
    static std::string vertShaderName() {
        static const char name[] = "foreground.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "foreground.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 1;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F};
        return formats;
    }
    static constexpr int numVertexAttributes = 1;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT };
        return types;
    }
};

unsigned char class_colors[22][3] = {{255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                              {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
                              {64, 0, 0}, {0, 64, 0}, {0, 0, 64}, {64, 64, 0}, {64, 0, 64}, {0, 64, 64}, 
                              {192, 0, 0}, {0, 192, 0}, {0, 0, 192}};


class Render
{
 public:

  Render() {};
  Render(std::string rig_specification_file, std::string model_file);
  ~Render() {};

  void setup(std::string rig_specification_file, std::string model_file);
  void setup_cameras(std::string rig_specification_file);
  void create_window();
  void destroy_window();
  float render(const float* data, const int* labels, const float* rois, int num_rois, int num_gt, int num_classes, 
               const float* poses_gt, const float* poses_pred, const float* poses_init, float* bottom_diff);
  void loadModels(std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(aiMesh* assimpMesh, std::string textureName, 
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);
  void feed_data(const float* data, const int* labels, pangolin::GlTexture & colorTex, pangolin::GlTexture & labelTex);

 private:
  df::Rig<double>* rig_;
  const df::CameraBase<double>* color_camera_;
  const df::CameraBase<double>* depth_camera_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;
  std::vector<std::string> texture_names_;

  // render
  df::GLRenderer<ForegroundRenderType>* renderer_;
  pangolin::OpenGlRenderState* rendererCam_;

  // pangoline views
  pangolin::View* gtView_;
  pangolin::View* poseView_;
  pangolin::View* colorView_;
  pangolin::View* labelView_;
  pangolin::View* multiView_;
};
