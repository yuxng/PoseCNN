#include "synthesize.hpp"

using namespace df;

void writeHalfPrecisionVertMap(const std::string filename, const float* vertMap, const int N) {

    std::cout << "writing " << filename << std::endl;

    std::vector<half> halfVertMap(N*3);
    for (int i=0; i<N; ++i) {
        halfVertMap[3*i  ] = half(vertMap[3*i]);
        halfVertMap[3*i+1] = half(vertMap[3*i+1]);
        halfVertMap[3*i+2] = half(vertMap[3*i+2]);
    }

    std::ofstream vertStream;
    vertStream.open(filename,std::ios_base::out | std::ios_base::binary);

    vertStream.write(reinterpret_cast<const char *>(halfVertMap.data()),halfVertMap.size()*sizeof(half));

    vertStream.close();

}

Synthesizer::Synthesizer(std::string model_file, std::string pose_file)
{
  model_file_ = model_file;
  pose_file_ = pose_file;
  counter_ = 0;
  setup_ = 0;
}

void Synthesizer::setup()
{
  int width = 640;
  int height = 480;
  create_window(width, height);

  loadModels(model_file_);
  std::cout << "loaded models" << std::endl;

  loadPoses(pose_file_);
  std::cout << "loaded poses" << std::endl;

  // create tensors
  labels_device_ = new df::ManagedDeviceTensor2<int>({width, height});
  intersection_device_ = new df::ManagedDeviceTensor2<int>({width, height});
  union_device_ = new df::ManagedDeviceTensor2<int>({width, height});
  vertex_map_device_ = new df::ManagedDeviceTensor2<Eigen::UnalignedVec4<float> >({width, height});

  setup_ = 1;
}

Synthesizer::~Synthesizer()
{
  destroy_window();
}

// create window
void Synthesizer::create_window(int width, int height)
{
  pangolin::CreateWindowAndBind("Synthesizer", 640, 480);

  gtView_ = &pangolin::Display("gt").SetAspect(640.0/480.);

  // create render
  renderer_ = new df::GLRenderer<df::CanonicalVertRenderType>(width, height);
}


void Synthesizer::destroy_window()
{
  pangolin::DestroyWindow("Synthesizer");
  delete renderer_;
}

// read the poses
void Synthesizer::loadPoses(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::string name;

  while ( std::getline (stream, name) )
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load poses
  const int num_models = model_names.size();
  poses_.resize(num_models);
  pose_nums_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    // cout lines
    int num_lines = 0;
    std::ifstream stream1(model_names[m]);
    std::string name;

    while ( std::getline (stream1, name) )
      num_lines++;
    stream1.close();
    pose_nums_[m] = num_lines;

    // allocate memory
    float* pose = (float*)malloc(sizeof(float) * num_lines * 7);

    // load data
    FILE* fp = fopen(model_names[m].c_str(), "r");
    for (int i = 0; i < num_lines * 7; i++)
      fscanf(fp, "%f", pose + i);
    fclose(fp);

    poses_[m] = pose;

    std::cout << model_names[m] << std::endl;
  }
}

// read the 3D models
void Synthesizer::loadModels(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::string name;

  while ( std::getline (stream, name) )
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }
  stream.close();

  // load meshes
  const int num_models = model_names.size();
  assimpMeshes_.resize(num_models);
  texture_names_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    assimpMeshes_[m] = loadTexturedMesh(model_names[m], texture_names_[m]);
    std::cout << texture_names_[m] << std::endl;
  }

  // buffers
  texturedVertices_.resize(num_models);
  canonicalVertices_.resize(num_models);
  texturedIndices_.resize(num_models);
  texturedCoords_.resize(num_models);
  texturedTextures_.resize(num_models);

  for (int m = 0; m < num_models; m++)
    initializeBuffers(m, assimpMeshes_[m], texture_names_[m], texturedVertices_[m], canonicalVertices_[m], texturedIndices_[m], texturedCoords_[m], texturedTextures_[m], true);
}

aiMesh* Synthesizer::loadTexturedMesh(const std::string filename, std::string & texture_name)
{
    const struct aiScene * scene = aiImportFile(filename.c_str(),0); //aiProcess_JoinIdenticalVertices);
    if (scene == 0) {
        throw std::runtime_error("error: " + std::string(aiGetErrorString()));
    }

    if (scene->mNumMeshes != 1) {
        const int nMeshes = scene->mNumMeshes;
        aiReleaseImport(scene);
        throw std::runtime_error("there are " + std::to_string(nMeshes) + " meshes in " + filename);
    }

    if (!scene->HasMaterials()) {
        throw std::runtime_error(filename + " has no materials");
    }

    std::cout << scene->mNumMaterials << " materials" << std::endl;

    std::string textureName = filename.substr(0,filename.find_last_of('/')+1);
    for (int i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial * material = scene->mMaterials[i];
        std::cout << "diffuse: " << material->GetTextureCount(aiTextureType_DIFFUSE) << std::endl;
        std::cout << "specular: " << material->GetTextureCount(aiTextureType_SPECULAR) << std::endl;
        std::cout << "ambient: " << material->GetTextureCount(aiTextureType_AMBIENT) << std::endl;
        std::cout << "shininess: " << material->GetTextureCount(aiTextureType_SHININESS) << std::endl;

        if (material->GetTextureCount(aiTextureType_DIFFUSE)) {

            aiString path;
            material->GetTexture(aiTextureType_DIFFUSE,0,&path);

            textureName = textureName + std::string(path.C_Str());

        }

    }

    aiMesh * assimpMesh = scene->mMeshes[0];

    if (!assimpMesh->HasTextureCoords(0)) {
        throw std::runtime_error("mesh does not have texture coordinates");
    }

    texture_name = textureName;
    return assimpMesh;
}


void Synthesizer::initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
  pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured)
{
    // std::cout << "loading vertices..." << std::endl;
    // std::cout << assimpMesh->mNumVertices << std::endl;
    vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    // canonical vertices
    std::vector<float3> canonicalVerts(assimpMesh->mNumVertices);
    std::memcpy(canonicalVerts.data(), assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float3));
    for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++)
        canonicalVerts[i].x += model_index;

    canonicalVertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    canonicalVertices.Upload(canonicalVerts.data(), assimpMesh->mNumVertices*sizeof(float3));

    // std::cout << "loading normals..." << std::endl;
    std::vector<uint3> faces3(assimpMesh->mNumFaces);
    for (std::size_t i = 0; i < assimpMesh->mNumFaces; i++) {
        aiFace & face = assimpMesh->mFaces[i];
        if (face.mNumIndices != 3) {
            throw std::runtime_error("not a triangle mesh");
        }
        faces3[i] = make_uint3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
    }

    indices.Reinitialise(pangolin::GlElementArrayBuffer,assimpMesh->mNumFaces*3,GL_UNSIGNED_INT,3,GL_STATIC_DRAW);
    indices.Upload(faces3.data(),assimpMesh->mNumFaces*sizeof(int)*3);

    if (is_textured)
    {
      std::cout << "loading texture from " << textureName << std::endl;
      texture.LoadFromFile(textureName);

      std::cout << "loading tex coords..." << std::endl;
      texCoords.Reinitialise(pangolin::GlArrayBuffer,assimpMesh->mNumVertices,GL_FLOAT,2,GL_STATIC_DRAW);

      std::vector<float2> texCoords2(assimpMesh->mNumVertices);
      for (std::size_t i = 0; i < assimpMesh->mNumVertices; ++i) {
          texCoords2[i] = make_float2(assimpMesh->mTextureCoords[0][i].x,1.0 - assimpMesh->mTextureCoords[0][i].y);
      }
      texCoords.Upload(texCoords2.data(),assimpMesh->mNumVertices*sizeof(float)*2);
    }
}


void Synthesizer::render(int width, int height, float fx, float fy, float px, float py, float znear, float zfar, 
              unsigned char* color, float* depth, float* vertmap, float* class_indexes, float *poses_return, float* centers_return,
              float* vertex_targets, float* vertex_weights, float weight)
{
  bool is_textured = true;
  int is_save = 0;

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);
  pangolin::OpenGlMatrixSpec projectionMatrix_reverse = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);

  // show gt pose
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  // sample the number of objects in the scene
  int num = irand(5, 9);
  int num_classes = pose_nums_.size();

  // sample object classes
  std::vector<int> class_ids(num);
  for (int i = 0; i < num; )
  {
    int class_id = irand(0, num_classes);
    int flag = 1;
    for (int j = 0; j < i; j++)
    {
      if(class_id == class_ids[j])
      {
        flag = 0;
        break;
      }
    }
    if (flag)
    {
      class_ids[i++] = class_id;
    }
  }

  if (class_indexes)
  {
    for (int i = 0; i < num; i++)
      class_indexes[i] = class_ids[i];
  }

  // store the poses
  std::vector<Sophus::SE3d> poses(num);
  double threshold = 0.2;

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    while(1)
    {
      // sample a pose
      int seed = irand(0, pose_nums_[class_id]);
      float* pose = poses_[class_id] + seed * 7;

      Eigen::Quaterniond quaternion(pose[0], pose[1], pose[2], pose[3]);
      Sophus::SE3d::Point translation(pose[4], pose[5], pose[6]);
      const Sophus::SE3d T_co(quaternion, translation);

      int flag = 1;
      for (int j = 0; j < i; j++)
      {
        Sophus::SE3d::Point T = poses[j].translation() - translation;
        double d = T.norm();
        if (d < threshold)
        {
          flag = 0;
          break;
        }
      }

      if (flag)
      {
        poses[i] = T_co;
        if (poses_return)
        {
          for (int j = 0; j < 7; j++)
            poses_return[i * 7 + j] = pose[j];
        }
        break;
      }
    }
  }

  // render vertmap
  std::vector<Eigen::Matrix4f> transforms(num);
  std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers(num);
  std::vector<pangolin::GlBuffer*> modelIndexBuffers(num);

  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];
    transforms[i] = poses[i].matrix().cast<float>();
    attributeBuffers[i].push_back(&texturedVertices_[class_id]);
    attributeBuffers[i].push_back(&canonicalVertices_[class_id]);
    modelIndexBuffers[i] = &texturedIndices_[class_id];
  }

  glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  renderer_->setProjectionMatrix(projectionMatrix_reverse);
  renderer_->render(attributeBuffers, modelIndexBuffers, transforms);

  glColor3f(1, 1, 1);
  gtView_->ActivateScissorAndClear();
  renderer_->texture(0).RenderToViewportFlipY();

  if (vertmap)
  {
    renderer_->texture(0).Download(vertmap, GL_RGB, GL_FLOAT);
    if (is_save)
    {
      std::string filename = std::to_string(counter_) + ".vertmap";
      writeHalfPrecisionVertMap(filename, vertmap, height*width);
    }

    // compute object 2D centers
    std::vector<float> center_x(num_classes, 0);
    std::vector<float> center_y(num_classes, 0);
    for (int i = 0; i < num; i++)
    {
      int class_id = class_ids[i];
      float tx = poses_return[i * 7 + 4];
      float ty = poses_return[i * 7 + 5];
      float tz = poses_return[i * 7 + 6];
      center_x[class_id] = fx * (tx / tz) + px;
      center_y[class_id] = fy * (ty / tz) + py;
    }

    if (centers_return)
    {
      for (int i = 0; i < num_classes; i++)
      {
        centers_return[2 * i] = center_x[i];
        centers_return[2 * i + 1] = center_y[i];
      }
    }

    // compute center regression targets and weights
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        float vx = vertmap[3 * (y * width + x)];
        if (std::isnan(vx))
          continue;
        int label = std::round(vx);
        // object center
        float cx = center_x[label];
        float cy = center_y[label];

        float rx = cx - x;
        float ry = cy - y;
        float norm = std::sqrt(rx * rx + ry * ry) + 1e-10;

        // assign value
        int offset = (label + 1) * 2 + 2 * (num_classes + 1) * (y * width + x);
        vertex_targets[offset] = rx / norm;
        vertex_weights[offset] = weight;

        offset = (label + 1) * 2 + 1 + 2 * (num_classes + 1) * (y * width + x);
        vertex_targets[offset] = ry / norm;
        vertex_weights[offset] = weight;
      }
    }
  }

  GLfloat lightpos0[] = {drand(-1., 1.), drand(-1., 1.), drand(1., 6.), 0.};
  GLfloat lightpos1[] = {drand(-1., 1.), drand(-1., 1.), drand(1., 6.), 0.};

  // render color image
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  for (int i = 0; i < num; i++)
  {
    int class_id = class_ids[i];

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = poses[i].cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos0);
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightpos1);

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id].Bind();
    texturedVertices_[class_id].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id].Unbind();
    texturedTextures_[class_id].Unbind();
    texturedVertices_[class_id].Unbind();
    texturedCoords_[class_id].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHT1);
    glDisable(GL_LIGHTING);
  }

  // read color image
  if (color)
  {
    glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, color);
    if (is_save)
    {
      cv::Mat C = cv::Mat(height, width, CV_8UC4, color);
      cv::Mat output;
      cv::flip(C, output, 0);
      std::string filename = std::to_string(counter_) + "_color.png";
      cv::imwrite(filename.c_str(), output);
    }
  }
  
  // read depth image
  if (depth)
  {
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth);

    if (is_save)
    {
      // write depth
      cv::Mat D = cv::Mat(height, width, CV_32FC1, depth);
      cv::Mat DD = cv::Mat(height, width, CV_16UC1);
      for (int x = 0; x < width; x++)
      { 
        for (int y = 0; y < height; y++)
        {
          if (D.at<float>(y, x) == 1)
            DD.at<short>(y, x) = 0;
          else
            DD.at<short>(y, x) = short(10000 * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * D.at<float>(y, x) - 1)));
        }
      }

      std::string filename = std::to_string(counter_) + "_depth.png";
      cv::Mat output;
      cv::flip(DD, output, 0);
      cv::imwrite(filename.c_str(), output);
    }
  }

  if (is_save)
  {
    std::string filename = std::to_string(counter_++);
    pangolin::SaveWindowOnRender(filename);
  }
  pangolin::FinishFrame();

  counter_++;
}

// get label lists
void Synthesizer::getLabels(const int* labelmap, std::vector<std::vector<int>>& labels, std::vector<int>& object_ids, int width, int height, int num_classes, int minArea)
{
  for(int i = 0; i < num_classes; i++)
    labels.push_back( std::vector<int>() );

  // for each pixel
  // #pragma omp parallel for
  for(int x = 0; x < width; x++)
  for(int y = 0; y < height; y++)
  {
    int label = labelmap[y * width + x];
    labels[label].push_back(y * width + x);
  }

  for(int i = 1; i < num_classes; i++)
  {
    if (labels[i].size() > minArea)
    {
      object_ids.push_back(i);
    }
  }
}


// get 3D bounding boxes
void Synthesizer::getBb3Ds(const float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
{
  // for each object
  bb3Ds.clear();
  for (int i = 1; i < num_classes; i++)
  {
    cv::Vec<float, 3> extent;
    extent(0) = extents[i * 3];
    extent(1) = extents[i * 3 + 1];
    extent(2) = extents[i * 3 + 2];

    bb3Ds.push_back(getBB3D(extent));
  }
}


inline cv::Point2f Synthesizer::getMode2D(jp::id_t objID, const cv::Point2f& pt, const float* vertmap, int width, int num_classes)
{
  int channel = 2 * objID;
  int offset = channel + 2 * num_classes * (pt.y * width + pt.x);

  jp::coord2_t mode;
  mode(0) = vertmap[offset];
  mode(1) = vertmap[offset + 1];

  return cv::Point2f(mode(0), mode(1));
}


inline bool Synthesizer::samplePoint2D(jp::id_t objID, std::vector<cv::Point2f>& eyePts, std::vector<cv::Point2f>& objPts, const cv::Point2f& pt2D, const float* vertmap, int width, int num_classes)
{
  cv::Point2f obj = getMode2D(objID, pt2D, vertmap, width, num_classes); // read out object coordinate

  eyePts.push_back(pt2D);
  objPts.push_back(obj);

  return true;
}


/**
 * @brief Creates a list of pose hypothesis (potentially belonging to multiple objects) which still have to be processed (e.g. refined).
 * 
 * The method includes all remaining hypotheses of an object if there is still more than one, or if there is only one remaining but it still needs to be refined.
 * 
 * @param hypMap Map of object ID to a list of hypotheses for that object.
 * @param maxIt Each hypotheses should be at least this often refined.
 * @return std::vector< Ransac3D::TransHyp*, std::allocator< void > > List of hypotheses to be processed further.
*/
std::vector<TransHyp*> Synthesizer::getWorkingQueue(std::map<jp::id_t, std::vector<TransHyp>>& hypMap, int maxIt)
{
  std::vector<TransHyp*> workingQueue;
      
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  for(int h = 0; h < it->second.size(); h++)
    if(it->second.size() > 1 || it->second[h].refSteps < maxIt) //exclude a hypothesis if it is the only one remaining for an object and it has been refined enough already
      workingQueue.push_back(&(it->second[h]));

  return workingQueue;
}


inline float Synthesizer::point2line(cv::Point2d x, cv::Point2f n, cv::Point2f p)
{
  float n1 = -n.y;
  float n2 = n.x;
  float p1 = p.x;
  float p2 = p.y;
  float x1 = x.x;
  float x2 = x.y;

  return fabs(n1 * (x1 - p1) + n2 * (x2 - p2)) / sqrt(n1 * n1 + n2 * n2);
}


inline void Synthesizer::countInliers2D(TransHyp& hyp, const float * vertmap, const std::vector<std::vector<int>>& labels, float inlierThreshold, int width, int num_classes, int pixelBatch)
{
  // reset data of last RANSAC iteration
  hyp.inlierPts2D.clear();
  hyp.inliers = 0;

  hyp.effPixels = 0; // num of pixels drawn
  hyp.maxPixels += pixelBatch; // max num of pixels to be drawn	

  int maxPt = labels[hyp.objID].size(); // num of pixels of this class
  float successRate = hyp.maxPixels / (float) maxPt; // probability to accept a pixel

  std::mt19937 generator;
  std::negative_binomial_distribution<int> distribution(1, successRate); // lets you skip a number of pixels until you encounter the next pixel to accept

  for(unsigned ptIdx = 0; ptIdx < maxPt;)
  {
    int index = labels[hyp.objID][ptIdx];
    cv::Point2d pt2D(index % width, index / width);
  
    hyp.effPixels++;
  
    // read out object coordinate
    cv::Point2d obj = getMode2D(hyp.objID, pt2D, vertmap, width, num_classes);

    // inlier check
    if(point2line(hyp.center, obj, pt2D) < inlierThreshold)
    {
      hyp.inlierPts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(obj, pt2D)); // store object coordinate - camera coordinate correspondence
      hyp.inliers++; // keep track of the number of inliers (correspondences might be thinned out for speed later)
    }

    // advance to the next accepted pixel
    if(successRate < 1)
      ptIdx += std::max(1, distribution(generator));
    else
      ptIdx++;
  }
}


inline void Synthesizer::updateHyp2D(TransHyp& hyp, int maxPixels)
{
  if(hyp.inlierPts2D.size() < 4) return;
  filterInliers2D(hyp, maxPixels); // limit the number of correspondences
      
  // data conversion
  cv::Point2d center = hyp.center;
  Hypothesis trans(center);	
	
  // recalculate pose
  trans.calcCenter(hyp.inlierPts2D);
  hyp.center = trans.getCenter();
}


inline void Synthesizer::filterInliers2D(TransHyp& hyp, int maxInliers)
{
  if(hyp.inlierPts2D.size() < maxInliers) return; // maximum number not reached, do nothing
      		
  std::vector<std::pair<cv::Point2d, cv::Point2d>> inlierPts; // filtered list of inlier correspondences
	
  // select random correspondences to keep
  for(unsigned i = 0; i < maxInliers; i++)
  {
    int idx = irand(0, hyp.inlierPts2D.size());
	    
    inlierPts.push_back(hyp.inlierPts2D[idx]);
  }
	
  hyp.inlierPts2D = inlierPts;
}


// Hough voting
void Synthesizer::estimateCenter(const int* labelmap, const float* vertmap, const float* extents, int height, int width, int num_classes, int preemptive_batch,
  float fx, float fy, float px, float py, float* outputs, float* gt_poses, int num_gt)
{
  //set parameters, see documentation of GlobalProperties
  int maxIterations = 10000000;
  float minArea = 400; // a hypothesis covering less projected area (2D bounding box) can be discarded (too small to estimate anything reasonable)
  float inlierThreshold3D = 0.5;
  int ransacIterations = 256;  // 256
  int preemptiveBatch = preemptive_batch;  // 1000
  int maxPixels = 1000;  // 1000
  int refIt = 8;  // 8

  // labels
  std::vector<std::vector<int>> labels;
  std::vector<int> object_ids;
  getLabels(labelmap, labels, object_ids, width, height, num_classes, minArea);
  std::cout << "read labels" << std::endl;

  // bb3Ds
  getBb3Ds(extents, bb3Ds_, num_classes);

  if (object_ids.size() == 0)
    return;
		
  // hold for each object a list of pose hypothesis, these are optimized until only one remains per object
  std::map<jp::id_t, std::vector<TransHyp>> hypMap;
	
  // sample initial pose hypotheses
  #pragma omp parallel for
  for(unsigned h = 0; h < ransacIterations; h++)
  for(unsigned i = 0; i < maxIterations; i++)
  {
    // camera coordinate - object coordinate correspondences
    std::vector<cv::Point2f> eyePts;
    std::vector<cv::Point2f> objPts;
	    
    // sample first point and choose object ID
    jp::id_t objID = object_ids[irand(0, object_ids.size())];

    if(objID == 0) continue;

    int pindex = irand(0, labels[objID].size());
    int index = labels[objID][pindex];
    cv::Point2f pt1(index % width, index / width);
    
    // sample first correspondence
    if(!samplePoint2D(objID, eyePts, objPts, pt1, vertmap, width, num_classes))
      continue;

    // sample other points in search radius, discard hypothesis if minimum distance constrains are violated
    pindex = irand(0, labels[objID].size());
    index = labels[objID][pindex];
    cv::Point2f pt2(index % width, index / width);

    if(!samplePoint2D(objID, eyePts, objPts, pt2, vertmap, width, num_classes))
      continue;

    // reconstruct camera
    std::vector<std::pair<cv::Point2d, cv::Point2d>> pts2D;
    for(unsigned j = 0; j < eyePts.size(); j++)
    {
      pts2D.push_back(std::pair<cv::Point2d, cv::Point2d>(
      cv::Point2d(objPts[j].x, objPts[j].y),
      cv::Point2d(eyePts[j].x, eyePts[j].y)
      ));
    }

    Hypothesis trans(pts2D);

    // center
    cv::Point2d center = trans.getCenter();
    
    // create a hypothesis object to store meta data
    TransHyp hyp(objID, center);
    
    #pragma omp critical
    {
      hypMap[objID].push_back(hyp);
    }

    break;
  }

  // create a list of all objects where hypptheses have been found
  std::vector<jp::id_t> objList;
  for(std::pair<jp::id_t, std::vector<TransHyp>> hypPair : hypMap)
  {
    objList.push_back(hypPair.first);
  }

  // create a working queue of all hypotheses to process
  std::vector<TransHyp*> workingQueue = getWorkingQueue(hypMap, refIt);
	
  // main preemptive RANSAC loop, it will stop if there is max one hypothesis per object remaining which has been refined a minimal number of times
  while(!workingQueue.empty())
  {
    // draw a batch of pixels and check for inliers, the number of pixels looked at is increased in each iteration
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
      countInliers2D(*(workingQueue[h]), vertmap, labels, inlierThreshold3D, width, num_classes, preemptiveBatch);
	    	    
    // sort hypothesis according to inlier count and discard bad half
    #pragma omp parallel for 
    for(unsigned o = 0; o < objList.size(); o++)
    {
      jp::id_t objID = objList[o];
      if(hypMap[objID].size() > 1)
      {
	std::sort(hypMap[objID].begin(), hypMap[objID].end());
	hypMap[objID].erase(hypMap[objID].begin() + hypMap[objID].size() / 2, hypMap[objID].end());
      }
    }
    workingQueue = getWorkingQueue(hypMap, refIt);
	    
    // refine
    #pragma omp parallel for
    for(int h = 0; h < workingQueue.size(); h++)
    {
      updateHyp2D(*(workingQueue[h]), maxPixels);
      workingQueue[h]->refSteps++;
    }
    
    workingQueue = getWorkingQueue(hypMap, refIt);
  }

  rois_.clear();
  for(auto it = hypMap.begin(); it != hypMap.end(); it++)
  {
    for(int h = 0; h < it->second.size(); h++)
    {
      std::cout << "Estimated Hypothesis for Object " << (int) it->second[h].objID << ":" << std::endl;

      jp::id_t objID = it->second[h].objID;
      cv::Point2d center = it->second[h].center;
      it->second[h].compute_width_height();
      outputs[4 * objID] = center.x;
      outputs[4 * objID + 1] = center.y;
      outputs[4 * objID + 2] = it->second[h].width_;
      outputs[4 * objID + 3] = it->second[h].height_;
    
      std::cout << "Inliers: " << it->second[h].inliers;
      std::printf(" (Rate: %.1f\%)\n", it->second[h].getInlierRate() * 100);
      std::cout << "Center " << center << std::endl;
      std::cout << "---------------------------------------------------" << std::endl;

      // roi (objID, x1, y1, x2, y2, rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z, score)
      // generate different pose hypotheses
      int interval = 8;
      for (int rx = 0; rx < interval; rx++)
      {
        for (int ry = 0; ry < interval; ry++)
        {
          for (int rz = 0; rz < interval; rz++)
          {
            cv::Vec<float, 12> roi;
            roi(0) = objID;
            // bounding box
            roi(1) = std::max(center.x - it->second[h].width_ / 2, 0.0);
            roi(2) = std::max(center.y - it->second[h].height_ / 2, 0.0);
            roi(3) = std::min(center.x + it->second[h].width_ / 2, double(width));
            roi(4) = std::min(center.y + it->second[h].height_ / 2, double(height));
            // 6D pose
            roi(5) = 2 * PI * rx / float(interval);
            roi(6) = 2 * PI * ry / float(interval);
            roi(7) = 2 * PI * rz / float(interval);
            // backproject the center
            roi(8) = (center.x - px) / fx;
            roi(9) = (center.y - py) / fy;
            roi(10) = 1.0;
            // score
            roi(11) = -1.0;
            rois_.push_back(roi);
          }
        }
      }

    }
  }

  int poseIterations = 10;
  estimatePose(labelmap, height, width, fx, fy, px, py, 0.25, 6.0, poseIterations, outputs);
  visualizePose(height, width, fx, fy, px, py, 0.25, 6.0, gt_poses, num_gt);
  usleep( 3000 * 1000 );
}


static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
{
  DataForOpt* dataForOpt = (DataForOpt*) data;
  // std::cout << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << " " << pose[4] << " " << pose[5] << std::endl;

  cv::Mat tvec(3, 1, CV_64F);
  cv::Mat rvec(3, 1, CV_64F);
      
  for(int i = 0; i < 6; i++)
  {
    if(i > 2) 
      tvec.at<double>(i-3, 0) = pose[i];
    else 
      rvec.at<double>(i, 0) = pose[i];
  }
	
  jp::cv_trans_t trans(rvec, tvec);

  // project the 3D bounding box according to the current pose
  cv::Rect bb2D = getBB2D(dataForOpt->imageWidth, dataForOpt->imageHeight, dataForOpt->bb3D, dataForOpt->camMat, trans);

  // project the 3D model to get the segmentation mask
  // convert pose
  cv::Vec<float, 3> rotation(pose[0], pose[1], pose[2]);
  cv::Mat rmat;
  cv::Rodrigues(rotation, rmat);
  rmat = rmat.t();
  Eigen::Matrix3f eigenT = Eigen::Map<Eigen::Matrix3f>( (float*)rmat.data );

  Eigen::Quaternionf quaternion(eigenT);
  Sophus::SE3f::Point translation(pose[3], pose[4], pose[5]);
  const Sophus::SE3f T_co(quaternion, translation);
  dataForOpt->transforms.clear();
  dataForOpt->transforms.push_back(T_co.matrix().cast<float>());

  glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
  dataForOpt->renderer->setProjectionMatrix(dataForOpt->projectionMatrix);
  dataForOpt->renderer->render(dataForOpt->attributeBuffers, dataForOpt->modelIndexBuffers, dataForOpt->transforms);

  glColor3f(1, 1, 1);
  dataForOpt->view->ActivateScissorAndClear();
  dataForOpt->renderer->texture(0).RenderToViewportFlipY();

  const pangolin::GlTextureCudaArray & vertTex = dataForOpt->renderer->texture(0);
  pangolin::CudaScopedMappedArray scopedArray(vertTex);
  cudaMemcpy2DFromArray(dataForOpt->vertex_map_device->data(), vertTex.width*4*sizeof(float), *scopedArray, 0, 0, vertTex.width*4*sizeof(float), vertTex.height, cudaMemcpyDeviceToDevice);

  // compute mask overlap
  /*
  int inter = 0;
  int all = 0;
  int x1 = std::round(std::min(bb2D.x, dataForOpt->bb2D.x));
  int x2 = std::round(std::max(bb2D.x + bb2D.width, dataForOpt->bb2D.x + dataForOpt->bb2D.width));
  int y1 = std::round(std::min(bb2D.y, dataForOpt->bb2D.y));
  int y2 = std::round(std::max(bb2D.y + bb2D.height, dataForOpt->bb2D.y + dataForOpt->bb2D.height));
  for (int x = x1; x <= x2; x++)
  {
    for (int y = y1; y <= y2; y++)
    {
      float vx = (*dataForOpt->vertex_map)(x, y)(0);
      int label = std::round(vx) + 1;
      int label_pred = dataForOpt->labelmap[y * dataForOpt->imageWidth + x];
      if (label == dataForOpt->classID || label_pred == dataForOpt->classID)
        all++;
      if (label == dataForOpt->classID && label_pred == dataForOpt->classID)
        inter++;
    }
  }
  */
  float IoU = iou(*dataForOpt->labels_device, *dataForOpt->intersection_device, *dataForOpt->union_device, *dataForOpt->vertex_map_device, dataForOpt->classID);
  float energy = -1 * IoU;

  return energy;
}


// pose refinement
void Synthesizer::estimatePose(const int* labelmap, int height, int width, float fx, float fy, float px, float py, float znear, float zfar, int poseIterations, float* outputs)
{
  if (setup_ == 0)
    setup();

  // copy labels to device
  df::ConstHostTensor2<int> labelImage({width, height}, labelmap);
  labels_device_->copyFrom(labelImage);

  DataForOpt data;
  data.imageHeight = height;
  data.imageWidth = width;
  data.projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);
  data.labelmap = labelmap;
  data.renderer = renderer_;
  data.view = gtView_;
  data.labels_device = labels_device_;
  data.intersection_device = intersection_device_;
  data.union_device = union_device_;
  data.vertex_map_device = vertex_map_device_;

  // camera matrix
  cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
  camMat(0, 0) = fx;
  camMat(1, 1) = fy;
  camMat(2, 2) = 1.f;
  camMat(0, 2) = px;
  camMat(1, 2) = py;
  data.camMat = camMat;

  // for each pose hypothesis
  int num = rois_.size();
  int id_prev = int(rois_[0](0));
  float max_energy = -100;
  for(int i = 0; i < num; i++)
  {
    cv::Vec<float, 12> roi = rois_[i];

    // 2D bounding box
    cv::Rect bb2D(roi(1), roi(2), roi(3) - roi(1), roi(4) - roi(2));

    // 3D bounding box
    int objID = int(roi(0));
    std::vector<cv::Point3f> bb3D = bb3Ds_[objID-1];

    // construct the data
    data.classID = objID;
    data.bb2D = bb2D;
    data.bb3D = bb3D;
    data.attributeBuffers.clear();
    std::vector<pangolin::GlBuffer *> buffer;
    buffer.push_back(&texturedVertices_[objID - 1]);
    buffer.push_back(&canonicalVertices_[objID - 1]);
    data.attributeBuffers.push_back(buffer);
    data.modelIndexBuffers.clear();
    data.modelIndexBuffers.push_back(&texturedIndices_[objID - 1]);

    // initialize pose
    std::vector<double> vec(6);
    vec[0] = roi(5);
    vec[1] = roi(6);
    vec[2] = roi(7);
    vec[3] = roi(8);
    vec[4] = roi(9);
    vec[5] = roi(10);

    // optimization
    float energy = poseWithOpt(vec, data, poseIterations);

    // convert pose to our format
    cv::Mat tvec(3, 1, CV_64F);
    cv::Mat rvec(3, 1, CV_64F);
      
    for(int i = 0; i < 6; i++)
    {
      if(i > 2) 
        tvec.at<double>(i-3, 0) = vec[i];
      else 
        rvec.at<double>(i, 0) = vec[i];
    }
	
    jp::cv_trans_t trans(rvec, tvec);
    jp::jp_trans_t pose = jp::cv2our(trans);

    // use the projected 3D box
    cv::Rect bb2D_proj = getBB2D(width, height, bb3D, camMat, trans);
    roi(1) = bb2D_proj.x;
    roi(2) = bb2D_proj.y;
    roi(3) = bb2D_proj.x + bb2D_proj.width;
    roi(4) = bb2D_proj.y + bb2D_proj.height;

    // update the pose hypothesis
    roi(5) = vec[0];
    roi(6) = vec[1];
    roi(7) = vec[2];
    roi(8) = vec[3];
    roi(9) = vec[4];
    roi(10) = vec[5];
    roi(11) = -energy;
    rois_[i] = roi;

    if (id_prev == objID)
    {
      if(roi(11) > max_energy)
      {
        outputs[4 * objID] = (roi(1) + roi(3)) / 2;
        outputs[4 * objID + 1] = (roi(2) + roi(4)) / 2;
        outputs[4 * objID + 2] = roi(3) - roi(1);
        outputs[4 * objID + 3] = roi(4) - roi(2);
        max_energy = roi(11);
      }
    }
    else
    {
      std::cout << "ID: " << id_prev << " score: " << max_energy << std::endl;
      outputs[4 * objID] = (roi(1) + roi(3)) / 2;
      outputs[4 * objID + 1] = (roi(2) + roi(4)) / 2;
      outputs[4 * objID + 2] = roi(3) - roi(1);
      outputs[4 * objID + 3] = roi(4) - roi(2);
      max_energy = roi(11);
    }
    id_prev = objID;
  }
}


double Synthesizer::poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations)
{
  // set up optimization algorithm (gradient free)
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 6); 

  // set optimization bounds 
  double rotRange = 22.5;
  rotRange *= PI / 180;
  double tRangeXY = 0.01;
  double tRangeZ = 0.5; // pose uncertainty is larger in Z direction
	
  std::vector<double> lb(6);
  lb[0] = vec[0]-rotRange; lb[1] = vec[1]-rotRange; lb[2] = vec[2]-rotRange;
  lb[3] = vec[3]-tRangeXY; lb[4] = vec[4]-tRangeXY; lb[5] = vec[5]-tRangeZ;
  opt.set_lower_bounds(lb);
      
  std::vector<double> ub(6);
  ub[0] = vec[0]+rotRange; ub[1] = vec[1]+rotRange; ub[2] = vec[2]+rotRange;
  ub[3] = vec[3]+tRangeXY; ub[4] = vec[4]+tRangeXY; ub[5] = vec[5]+tRangeZ;
  opt.set_upper_bounds(ub);
      
  // configure NLopt
  opt.set_min_objective(optEnergy, &data);
  opt.set_maxeval(iterations);

  // run optimization
  double energy;
  nlopt::result result = opt.optimize(vec, energy);

  // std::cout << "IoU after optimization: " << -energy << std::endl;
   
  return energy;
}


void Synthesizer::visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, float* gt_poses, int num_gt)
{
  if (setup_ == 0)
    setup();

  int num = rois_.size();
  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);

  // find the closest poses hypthosis for each gt
  std::vector<float> scores(num_gt, -100);
  std::vector<Sophus::SE3f> poses(num_gt);
  for (int i = 0; i < num_gt; i++)
  {
    for (int j = 0; j < num; j++)
    {
      cv::Vec<float, 12> roi = rois_[j];
      if (int(roi[0]) != int(gt_poses[i * 8]))
        continue;

      // convert pose
      cv::Vec<float, 3> rotation(roi[5], roi[6], roi[7]);
      cv::Mat rmat;
      cv::Rodrigues(rotation, rmat);
      rmat = rmat.t();
      Eigen::Matrix3f eigenT = Eigen::Map<Eigen::Matrix3f>( (float*)rmat.data );

      Eigen::Quaternionf quaternion(eigenT);
      Sophus::SE3f::Point translation(roi[8], roi[9], roi[10]);
      const Sophus::SE3f T_co(quaternion, translation);

      float score = roi[11];
      if (score > scores[i])
      {
        scores[i] = score;
        poses[i] = T_co;
      }
    }
    std::cout << "ID: " << int(gt_poses[i * 8]) << " score: " << scores[i] << std::endl;
  }

  /*
  for (int i = 0; i < num_gt; i++)
  {
    Eigen::Quaternionf quaternion(gt_poses[i * 8 + 1], gt_poses[i * 8 + 2], gt_poses[i * 8 + 3], gt_poses[i * 8 + 4]);
    Sophus::SE3f::Point translation(gt_poses[i * 8 + 5], gt_poses[i * 8 + 6], gt_poses[i * 8 + 7]);
    const Sophus::SE3f T_co(quaternion, translation);
    poses[i] = T_co;
    std::cout << i << " " << gt_poses[i * 8 + 1] << " " << gt_poses[i * 8 + 2] << " " << gt_poses[i * 8 + 3] << " " << gt_poses[i * 8 + 4] << std::endl;
  }
  */

  // render color image
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  for (int i = 0; i < num_gt; i++)
  {
    int class_id = int(gt_poses[i * 8]) - 1;

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = poses[i].cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id].Bind();
    texturedVertices_[class_id].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id].Unbind();
    texturedTextures_[class_id].Unbind();
    texturedVertices_[class_id].Unbind();
    texturedCoords_[class_id].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

  }
  pangolin::FinishFrame();
}


int main(int argc, char** argv) 
{
  Synthesizer Synthesizer(argv[1], argv[2]);
  Synthesizer.setup();

  // camera parameters
  int width = 640;
  int height = 480;
  float fx = 1066.778, fy = 1067.487, px = 312.9869, py = 241.3109, zfar = 6.0, znear = 0.25;

  while (!pangolin::ShouldQuit()) 
  {
    clock_t start = clock();    

    Synthesizer.render(width, height, fx, fy, px, py, znear, zfar, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1.0);

    clock_t stop = clock();    
    double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Time elapsed in ms: %f\n", elapsed);
  }
}
