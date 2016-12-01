namespace df {

class KinectFusion
{
 public:
  KinectFusion(std::string rig_specification_file);
  ~KinectFusion() {};

  void solve_pose(float* pose);
  void fuse_depth();
  void extract_surface();
  void render();
  void draw();
  void back_project();
  void feed_data(unsigned char* depth, unsigned char* color, int width, int height);
  void reset();
};

}
