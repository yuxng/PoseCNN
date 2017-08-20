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
#include <Eigen/Geometry>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <OpenEXR/half.h>

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

#include "types.h"
#include "ransac.h"
#include "Hypothesis.h"
#include "detection.h"
#include "thread_rand.h"
#include "iou.h"

typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vec3;

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

struct DataForOpt
{
  int num_roi;
  int classID;

  std::vector<Eigen::Matrix4f> transforms;
  std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers;
  std::vector<pangolin::GlBuffer*> modelIndexBuffers;
  df::GLRenderer<df::CanonicalVertRenderType>* renderer;
  pangolin::View* view;

  df::ManagedDeviceTensor2<int>* labels_device;
  df::ManagedDeviceTensor2<int>* intersection_device;
  df::ManagedDeviceTensor2<int>* union_device;
  df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* vertex_map_device;
};

class Synthesizer
{
 public:

  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer();

  void setup();
  void create_window(int width, int height);
  void destroy_window();
  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float* class_indexes, float *poses_return, float* centers_return,
              float* vertex_targets, float* vertex_weights, float weight);
  void render_one(int which_class, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float *poses_return, float* centers_return, float* extents);
  void loadModels(std::string filename);
  void loadPoses(const std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors,
    pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);

  jp::jp_trans_t quat2our(const Sophus::SE3d T_co);

  // hough voting
  void estimateCenter(const int* labelmap, const float* vertmap, const float* extents, int height, int width, int num_classes, int preemptive_batch,
                      float fx, float fy, float px, float py, float* outputs, float* gt_poses, int num_gt);
  inline void filterInliers2D(TransHyp& hyp, int maxInliers);
  inline void updateHyp2D(TransHyp& hyp, int maxPixels);
  inline void countInliers2D(TransHyp& hyp, const float * vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes, int pixelBatch);
  inline float point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p);
  std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt);
  inline bool samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, const cv::Point2f& pt2D, const float* vertmap, int width, int num_classes);
  inline cv::Point2f getMode2D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, int width, int num_classes);
  void getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
  void getLabels(const int* label_map, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea);

  // pose refinement with bounding box
  void estimatePose(const int* labelmap, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                    int num_roi, float* rois, float* poses, float* outputs, int poseIterations);
  double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations);

  void visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, float* rois, float* outputs, int num_roi);

 private:
  int counter_;
  int setup_;
  std::string model_file_, pose_file_;

  // poses
  std::vector<float*> poses_;
  std::vector<int> pose_nums_;
  std::vector<bool> is_textured_;

  // rois
  std::vector<std::vector<cv::Vec<float, 12> > > rois_;

  // 3D bounding boxes
  std::vector<std::vector<cv::Point3f>> bb3Ds_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;

  // pangoline views
  pangolin::View* gtView_;

  // buffers
  std::vector<pangolin::GlBuffer> texturedVertices_;
  std::vector<pangolin::GlBuffer> canonicalVertices_;
  std::vector<pangolin::GlBuffer> vertexColors_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<df::CanonicalVertRenderType>* renderer_;

  // device tensors
  df::ManagedDeviceTensor2<int>* labels_device_;
  df::ManagedDeviceTensor2<int>* intersection_device_;
  df::ManagedDeviceTensor2<int>* union_device_;
  df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* vertex_map_device_;

};
