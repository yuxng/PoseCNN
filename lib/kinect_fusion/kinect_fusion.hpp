#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/video/video.h>
#include <getopt.h>
#include <df/camera/camera.h>
#include <df/camera/cameraFactory.h>
#include <df/camera/poly3.h>
#include <df/camera/rig.h>
#include <df/image/backprojection.h>
#include <df/fusion/fusion.h>
#include <df/optimization/icp.h>
#include <df/prediction/glRender.h>
#include <df/prediction/glRenderTypes.h>
#include <df/prediction/raycast.h>
#include <df/surface/marchingCubes.h>
#include <df/surface/normals.h>
#include <df/surface/color.h>
#include <df/transform/rigid.h>
#include <df/util/args.h>
#include <df/util/cudaHelpers.h>
#include <df/util/fpsCounter.h>
#include <df/util/glHelpers.h>
#include <df/util/globalTimer.h>
#include <df/util/pangolinHelpers.h>
#include <df/util/tensor.h>
#include <df/voxel/tsdf.h>
#include <df/voxel/voxelGrid.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/probability.h>

namespace df {

typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vec3;
typedef Eigen::Matrix<float,10,1,Eigen::DontAlign> Vec;
typedef Eigen::Matrix<int,3,1,Eigen::DontAlign> Vec3i;
typedef Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> Vec3uc;

class KinectFusion
{
 public:
  KinectFusion(std::string rig_specification_file);
  ~KinectFusion() {};

  void setup_cameras(std::string rig_specification_file);
  void create_window();
  void create_tensors();

  void solve_pose(float* pose_worldToLive, float* pose_liveToWorld);
  void fuse_depth();
  void extract_surface(int* labels_return);
  void render();
  void draw(std::string filename, int flag);
  void back_project();
  void feed_data(unsigned char* depth, unsigned char* color, int width, int height, float factor);
  void feed_label(unsigned char* im_label, float* probability, unsigned char* colors);
  void reset();
  void set_voxel_grid(float voxelGridOffsetX, float voxelGridOffsetY, float voxelGridOffsetZ, float voxelGridDimX, float voxelGridDimY, float voxelGridDimZ);
  void save_model(std::string filename);

  ManagedTensor<2, float>* depth_map() { return depth_map_; };
  pangolin::GlTexture* color_texture() { return colorTex_; };
  pangolin::GlTexture* depth_texture() { return depthTex_; };
  pangolin::GlTexture* label_texture() { return labelTex_; };

  void renderModel(pangolin::GlBufferCudaPtr & vertBuffer, pangolin::GlBufferCudaPtr & normBuffer, pangolin::GlBufferCudaPtr & indexBuffer, pangolin::GlBufferCudaPtr & colorBuffer);
  void renderModel(pangolin::GlBufferCudaPtr & vertBuffer, pangolin::GlBufferCudaPtr & normBuffer, pangolin::GlBufferCudaPtr & indexBuffer);

 private:

  // cameras
  Rig<double>* rig_;
  pangolin::OpenGlRenderState* colorCamState_;
  pangolin::OpenGlRenderState* depthCamState_;
  const CameraBase<double>* color_camera_;
  const CameraBase<double>* depth_camera_;
  Eigen::Matrix3f colorKinv_, depthKinv_;
  Sophus::SE3d T_dc_;

  // depths
  float depth_factor_;
  float depth_cutoff_;
  ManagedTensor<2, float>* depth_map_;
  ManagedTensor<2, float, DeviceResident>* depth_map_device_;

  // probability
  ManagedDeviceTensor2<Vec>* probability_map_device_;

  // class colors
  ManagedDeviceTensor1<Vec3uc>* class_colors_device_;

  // color
  ManagedHostTensor2<Vec3>* color_map_;
  ManagedDeviceTensor2<Vec3>* color_map_device_;

  // labels
  ManagedHostTensor2<int>* labels_;
  ManagedDeviceTensor2<int>* labels_device_;

  ManagedHostTensor2<Vec3uc>* label_colors_;
  ManagedDeviceTensor2<Vec3uc>* label_colors_device_;

  // 3D points
  ManagedHostTensor2<Vec3>* vertex_map_;
  ManagedDeviceTensor2<Vec3>* vertex_map_device_;

  // predicted vertices and normals
  ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_verts_;
  ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_normals_;

  ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* predicted_verts_device_;
  ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >* predicted_normals_device_;

  // in extract surface
  ManagedTensor<2, float, DeviceResident>* dVertices_;
  ManagedTensor<2, float, DeviceResident>* dWeldedVertices_;
  ManagedTensor<1, int, DeviceResident>* dIndices_;
  ManagedDeviceTensor1<Eigen::UnalignedVec3<float> >* dNormals_;
  // ManagedTensor<2, unsigned char, DeviceResident>* dColors_;
  ManagedDeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> >* dColors_;
  uint numUniqueVertices_;

  // voxels
  ManagedTensor<3, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident>* voxel_data_;
  DeviceVoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >* voxel_grid_;

  // ICP
  RigidTransformer<float>* transformer_;

  // rendering
  pangolin::GlBufferCudaPtr* vertBuffer_;
  pangolin::GlBufferCudaPtr* normBuffer_;
  pangolin::GlBufferCudaPtr* indexBuffer_;
  pangolin::GlBufferCudaPtr* colorBuffer_;

  // render
  GLRenderer<VertAndNormalRenderType>* renderer_;

  // draw
  pangolin::View* allView_;
  pangolin::View* disp3d_;
  pangolin::View* colorView_;
  pangolin::View* depthView_;
  pangolin::View* labelView_;
  pangolin::GlTexture* colorTex_;
  pangolin::GlTexture* depthTex_;
  pangolin::GlTexture* labelTex_;
};

}

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}
