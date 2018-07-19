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
#include <OpenEXR/half.h>

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

#include "types.h"
#include "ransac.h"
#include "Hypothesis.h"
#include "detection.h"
#include "thread_rand.h"
#include "iou.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
namespace np = boost::python::numpy;

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

unsigned char class_colors[22][3] = {{255, 255, 255}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255},
                              {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128},
                              {64, 0, 0}, {0, 64, 0}, {0, 0, 64}, {64, 64, 0}, {64, 0, 64}, {0, 64, 64}, 
                              {192, 0, 0}, {0, 192, 0}, {0, 0, 192}};

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

  TransHyp* hyp;
  cv::Mat* camMat;
};

class Synthesizer
{
 public:

  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer();

  void setup(int width, int height);
  void init_rand(unsigned seed);
  void create_window(int width, int height);
  void destroy_window();
  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, float tnear, float tfar, 
              float* color, float* depth, float* vertmap, float* class_indexes, float *poses_return, float* centers_return,
              bool is_sampling, bool is_sampling_pose);
  void render_python(int width, int height, np::ndarray const & parameters, 
    np::ndarray const & color, np::ndarray const & depth, np::ndarray const & vertmap, np::ndarray const & class_indexes, 
    np::ndarray const & poses_return, np::ndarray const & centers_return, bool is_sampling, bool is_sampling_pose);

  void render_poses_python(int num, int channel, int width, int height, np::ndarray const & parameters, 
    np::ndarray const & color, np::ndarray const & poses);
  void render_poses(int num, int channel, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
    unsigned char* color, float *poses_input);

  void render_poses_color_python(int num, int channel, int width, int height, np::ndarray const & parameters, 
    np::ndarray const & color, np::ndarray const & poses);
  void render_poses_color(int num, int channel, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
    unsigned char* color, float *poses_input);

  void render_one(int which_class, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              float* color, float* depth, float* vertmap, float *poses_return, float* centers_return, float* extents);
  void render_one_python(int which_class, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              np::ndarray& color, np::ndarray& depth, np::ndarray& vertmap, np::ndarray& poses_return, np::ndarray& centers_return, np::ndarray& extents);

  void loadModels(std::string filename);
  void loadPoses(const std::string filename);
  aiMesh* loadTexturedMesh(const std::string filename, std::string & texture_name);
  void initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
    pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
    pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured);

  jp::jp_trans_t quat2our(const Sophus::SE3d T_co);

  // pose refinement with ICP
  void solveICP(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                float factor, int num_roi, int channel_roi, const float* rois, const float* poses, float* outputs, float* outputs_icp, float maxError);

  void icp_python(np::ndarray& labelmap, np::ndarray& depth, np::ndarray& parameters, 
    int height, int width, int num_roi, int channel_roi, 
    np::ndarray& rois, np::ndarray& poses, np::ndarray& outputs, np::ndarray& outputs_icp, float   maxError);

  void visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, const float* rois, float* outputs, int num_roi, int channel_roi);
  double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations);
  void refinePose(int width, int height, int objID, float znear, float zfar,
                  const int* labelmap, DataForOpt data, df::Poly3CameraModel<float> model, Sophus::SE3f & T_co, int iterations, float maxError, int is_icp);

  // pose estimation with color
  void estimatePose2D(const int* labelmap, const float* vertmap, const float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float* output);
  inline void filterInliers2D(TransHyp& hyp, int maxInliers);
  inline void updateHyp2D(TransHyp& hyp, const cv::Mat& camMat, int imgWidth, int imgHeight, const std::vector<cv::Point3f>& bb3D, int maxPixels);
  inline void countInliers2D(TransHyp& hyp, const cv::Mat& camMat, const std::vector<std::vector<int>>& labels, const float* vertmap,
      const float* extents, float inlierThreshold, int width, int num_classes, int pixelBatch);
  inline float point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p);
  std::vector<TransHyp*> getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt);
  inline bool samplePoint2D(jp::id_t objID, int width, int num_classes, std::vector<cv::Point2f>& pts2D, 
    std::vector<cv::Point3f>& pts3D, const cv::Point2f& pt2D, const float* vertmap, const float* extents, float minDist2D, float minDist3D);
  void getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes);
  void getLabels(const int* label_map, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea);

  // pose estimation with depth
  void estimatePose3D(const int* labelmap, unsigned char* rawdepth, const float* vertmap, const float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float depth_factor, float* output);
  inline void updateHyp3D(TransHyp& hyp, const cv::Mat& camMat, int imgWidth, int imgHeight, const std::vector<cv::Point3f>& bb3D, int maxPixels);
  inline void filterInliers3D(TransHyp& hyp, int maxInliers);
  inline void countInliers3D(TransHyp& hyp, const std::vector<std::vector<int>>& labels, const float* vertmap, const float* extents, const jp::img_coord_t& eyeData,float inlierThreshold, int width, int num_classes, int pixelBatch);
  inline bool samplePoint3D(jp::id_t objID, int width, int num_classes, std::vector<cv::Point3f>& eyePts, std::vector<cv::Point3f>& objPts, const cv::Point2f& pt2D,
      const float* vertmap, const float* extents, const jp::img_coord_t& eyeData, float minDist3D);
  inline cv::Point3f getMode3D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, const float* extents, int width, int num_classes);
  template<class T> inline double getMinDist(const std::vector<T>& pointSet, const T& point);
  void getEye(unsigned char* rawdepth, jp::img_coord_t& img, jp::img_depth_t& img_depth, int width, int height, float fx, float fy, float px, float py, float depth_factor);
  jp::coord3_t pxToEye(int x, int y, jp::depth_t depth, float fx, float fy, float px, float py, float depth_factor);

  double refineWithOpt(TransHyp& hyp, cv::Mat& camMat, int iterations, int is_3D);
  inline double pointLineDistance(const cv::Point3f& pt1, const cv::Point3f& pt2, const cv::Point3f& pt3);

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
  std::vector<Eigen::Quaterniond> poses_uniform_;
  int pose_index_;

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
  std::vector<pangolin::GlBuffer> vertexNormals_;
  std::vector<pangolin::GlBuffer> texturedIndices_;
  std::vector<pangolin::GlBuffer> texturedCoords_;
  std::vector<pangolin::GlTexture> texturedTextures_;

  df::GLRenderer<df::CanonicalVertRenderType>* renderer_;
  df::GLRenderer<df::CanonicalVertAndTextureRenderType>* renderer_texture_;
  df::GLRenderer<df::CanonicalVertAndColorRenderType>* renderer_color_;
  df::GLRenderer<df::VertAndNormalRenderType>* renderer_vn_;
};

using namespace boost::python;
BOOST_PYTHON_MODULE(libsynthesizer)
{
  np::initialize();
  class_<Synthesizer>("Synthesizer", init<std::string, std::string>())
    .def("setup", &Synthesizer::setup)
    .def("init_rand", &Synthesizer::init_rand)
    .def("render_one_python", &Synthesizer::render_one_python)
    .def("render_python", &Synthesizer::render_python)
    .def("render_poses_python", &Synthesizer::render_poses_python)
    .def("render_poses_color_python", &Synthesizer::render_poses_color_python)
    .def("icp_python", &Synthesizer::icp_python)
  ;
}
