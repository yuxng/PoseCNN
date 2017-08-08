class Synthesizer
{
 public:
  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer() {};
  void setup();

  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
    unsigned char* color, float* depth, float* vertmap, float* class_indexes, float* poses, float* centers,
    float* vertex_targets, float* vertex_weights, float weight);

  void estimateCenter(const int* labelmap, const float* vertmap, const float* extents, int height, int width, int num_classes, int preemptive_batch,
    float fx, float fy, float px, float py, float* outputs,  float* gt_poses, int num_gt);

  void estimatePose(const int* labelmap, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, 
    int num_roi, float* rois, float* poses, float* outputs, int poseIterations);
};
