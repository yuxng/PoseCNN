class Synthesizer
{
 public:
  Synthesizer(std::string model_file, std::string pose_file);
  ~Synthesizer() {};
  void render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
    unsigned char* color, float* depth, float* vertmap, float* class_indexes, float* poses,
    float* vertex_targets, float* vertex_weights, float weight);
};
