namespace df {

class KinectFusion
{
 public:
  KinectFusion(std::string rig_specification_file);
  ~KinectFusion() {};

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
};

}
