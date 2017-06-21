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

#include "opencv2/opencv.hpp"

#define GL_GLEXT_PROTOTYPES
#include "GL/gl.h"
#include "GL/glext.h"
#include "GL/osmesa.h"

#include <cuda_runtime.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

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

template<typename P> inline
void OpenGlMatrix(const Eigen::Matrix<P,4,4>& mat, float* m)
{
    for(int r=0; r<4; ++r ) {
        for(int c=0; c<4; ++c ) {
            m[c*4+r] = mat(r,c);
        }
    }
}

unsigned char class_colors[22][3] = {{255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                              {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
                              {64, 0, 0}, {0, 64, 0}, {0, 0, 64}, {64, 64, 0}, {64, 0, 64}, {0, 64, 64}, 
                              {192, 0, 0}, {0, 192, 0}, {0, 0, 192}};


typedef struct
{
  int num_vertices;
  int num_faces;
  void* vertices;
  void* faces; 
}MyModel;

class Render
{
 public:

  Render() {};
  Render(std::string model_file);
  ~Render();

  void setup(std::string model_file);
  float render(const float* data, const int* labels, const float* rois, int num_rois, int num_gt, int num_classes, int width, int height,
               const float* poses_gt, const float* poses_pred, const float* poses_init, float* bottom_diff, const float* meta_data, int num_meta_data);
  void loadModels(const std::string filename);
  void loadTexturedMesh(const std::string filename, std::string & texture_name, MyModel* model);
  void initializeBuffers(MyModel* model, std::string textureName, GLuint vertexbuffer, GLuint indexbuffer);
  void ProjectionMatrixRDF_TopLeft(float* m, int w, int h, float fu, float fv, float u0, float v0, float zNear, float zFar );
  void write_ppm(const char *filename, const GLubyte *buffer, int width, int height);
  void print_matrix(float *m);

 private:
  int counter_;

  // 3D models
  std::vector<MyModel*> models_;
  std::vector<std::string> texture_names_;
};
