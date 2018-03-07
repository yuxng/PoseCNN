class Synthesizer
{
 public:
  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer() {};
  void setup(int width, int height);

  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
    unsigned char* color, float* depth, float* vertmap, float* class_indexes, float* poses, float* centers,
    float* vertex_targets, float* vertex_weights, float weight);

  void render_one(int which_class, int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float *poses_return, float* centers_return, float* extents);

  void estimatePose2D(const int* labelmap, const float* vertmap, const float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float* output);

  void solveICP(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
                float factor, int num_roi, int channel_roi, const float* rois, const float* poses, float* outputs, float* outputs_icp, float maxError);

  void estimatePose3D(const int* labelmap, unsigned char* rawdepth, const float* vertmap, const float* extents,
        int width, int height, int num_classes, float fx, float fy, float px, float py, float depth_factor, float* output);
};
