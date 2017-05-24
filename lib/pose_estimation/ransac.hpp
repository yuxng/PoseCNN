namespace jp {

class Ransac3D
{
 public:
  Ransac3D();
  ~Ransac3D() {};

  float estimatePose(unsigned char* rawdepth, float* probability, float* vertmap, float* extents,
    int width, int height, int num_classes, float fx, float fy, float px, float py, float depth_factor, float* output);

  float estimateCenter(float* probability, float* vertmap, int width, int height, int num_classes, float* output);
};

}
