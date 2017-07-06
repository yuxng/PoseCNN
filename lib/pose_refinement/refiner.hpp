class Refiner
{
 public:
  Refiner(std::string model_file);
  ~Refiner() {};

  void setup(std::string model_file);

  void render(unsigned char* data, unsigned char* labels, float* rois, int num_rois, int num_gt, int width, int height, int num_classes,
                    float* poses_gt, float* poses_pred, float fx, float fy, float px, float py, float* extents, float* poses_new, int is_save);
};
