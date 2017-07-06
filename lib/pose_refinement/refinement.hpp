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
#include <nlopt.hpp>
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

struct DataForOpt
{
  int width, height;
  cv::Rect bb2D;
  std::vector<cv::Point3f> bb3D;
  cv::Mat_<float> camMat;
  pangolin::OpenGlMatrixSpec projectionMatrix;
  pangolin::GlBuffer* texturedVertices;
  pangolin::GlBuffer* texturedIndices;
  cv::Mat* gt_mask;
  pangolin::View* view;
  df::GLRenderer<ForegroundRenderType>* renderer;
};

static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data);
double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations);
inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2);
int clamp(int val, int min_val, int max_val);
inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& RT);

class Refiner
{
 public:

  Refiner() {};
  Refiner(std::string model_file);
  ~Refiner();

  void setup(std::string model_file);
  void create_window(int width, int height);
  void destroy_window();
  void render(unsigned char* data, unsigned char* labels, float* rois, int num_rois, int num_gt, int width, int height, int num_classes,
                    float* poses_gt, float* poses_pred, float fx, float fy, float px, float py, float* extents, float* poses_new, int is_save);
  void loadModels(std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(aiMesh* assimpMesh, std::string textureName, 
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);
  void feed_data(int width, int height, unsigned char* data, unsigned char* labels, pangolin::GlTexture & colorTex, pangolin::GlTexture & labelTex);

  void refine(unsigned char* labels, float* rois, int num_rois, int width, int height, int num_classes,
                    float* poses_pred, float fx, float fy, float px, float py, float* extents, float* poses_new);

  void getBb3Ds(float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
  inline std::vector<cv::Point3f> getBB3D(const cv::Vec<float, 3>& extent);

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
  pangolin::View* maskView_;
  pangolin::View* multiView_;

  // buffers
  std::vector<pangolin::GlBuffer> texturedVertices_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<ForegroundRenderType>* renderer_;
};
