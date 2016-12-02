namespace df {

class KinectFusion
{
 public:
  KinectFusion(std::string rig_specification_file);
  ~KinectFusion() {};

  void solve_pose(float* pose_worldToLive, float* pose_liveToWorld);
  void fuse_depth();
  void extract_surface();
  void render();
  void draw();
  void back_project();
  void feed_data(unsigned char* depth, unsigned char* color, int width, int height);
  void reset();
  void set_voxel_grid(float voxelGridOffsetX, float voxelGridOffsetY, float voxelGridOffsetZ, float voxelGridDimX, float voxelGridDimY, float voxelGridDimZ);
};

}
