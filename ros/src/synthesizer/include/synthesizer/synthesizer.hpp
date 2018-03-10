#pragma once

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
#include <assimp/Importer.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/geometry.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/crh.h>
#include <pcl/kdtree/kdtree_flann.h>

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
#include <df/optimization/icp.h>

#include <ros/ros.h>
#include <geometry_msgs/Point32.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
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

struct DataForOpt
{
  int width, height, objID;

  std::vector<pangolin::GlBuffer*> attributeBuffers;
  std::vector<pangolin::GlBuffer*> modelIndexBuffers;
  df::GLRenderer<df::VertAndNormalRenderType>* renderer;
  pangolin::View* view;
  df::Poly3CameraModel<float>* model;

  const int* labelmap;
  std::vector<int>* label_indexes;
  Eigen::Vector2f depthRange;
  df::ManagedHostTensor2<Vec3>* vertex_map;
  df::ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_verts;
  df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* predicted_verts_device;
};

class Synthesizer
{
 public:

  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer();

  void setup(int width, int height);
  void create_window(int width, int height);
  void destroy_window();
  void loadModels(std::string filename);
  void loadPoses(const std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
    pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);

  // pose refinement with ICP
  void refineDistance(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                float factor, int num_roi, int channel_roi, const float* rois, const float* poses, float* outputs, float* outputs_icp,                 
                std::vector<std::vector<geometry_msgs::Point32> >& output_points, float maxError);
  void solveICP(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                float factor, int num_roi, int channel_roi, const float* rois, const float* poses, float* outputs, float* outputs_icp, float maxError);
  void visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, const float* rois, float* outputs, int num_roi, int channel_roi);
  double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations);
  void refinePose(int width, int height, int objID, float znear, float zfar,
                  const int* labelmap, DataForOpt data, df::Poly3CameraModel<float> model, Sophus::SE3f & T_co, int iterations, float maxError, int is_icp);

 private:
  int counter_;
  int setup_;
  std::string model_file_, pose_file_;

  df::ManagedDeviceTensor2<int>* labels_device_;

  // depths
  float depth_factor_;
  float depth_cutoff_;
  std::vector<int> label_indexes_;
  df::ManagedTensor<2, float>* depth_map_;
  df::ManagedTensor<2, float, df::DeviceResident>* depth_map_device_;

  // 3D points
  df::ManagedDeviceTensor2<Vec3>* vertex_map_device_;
  df::ManagedHostTensor2<Vec3>* vertex_map_;
  df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* predicted_verts_device_;
  df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* predicted_normals_device_;
  df::ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_verts_;
  df::ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_normals_;

  // poses
  std::vector<float*> poses_;
  std::vector<int> pose_nums_;
  std::vector<bool> is_textured_;

  // rois
  std::vector<std::vector<cv::Vec<float, 12> > > rois_;

  // 3D models
  std::vector<aiMesh*> assimpMeshes_;

  // pangoline views
  pangolin::View* gtView_;

  // buffers
  std::vector<pangolin::GlBuffer> texturedVertices_;
  std::vector<pangolin::GlBuffer> canonicalVertices_;
  std::vector<pangolin::GlBuffer> vertexColors_;
  std::vector<pangolin::GlBuffer> vertexNormals_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<df::CanonicalVertRenderType>* renderer_;
  df::GLRenderer<df::VertAndNormalRenderType>* renderer_vn_;
};
