#include "refinement.hpp"

using namespace df;

Refiner::Refiner(std::string model_file)
{
  counter_ = 0;
  create_window(640.0, 480.0);

  loadModels(model_file);
  std::cout << "loaded models" << std::endl;
}


void Refiner::setup(std::string model_file)
{
  counter_ = 0;
  loadModels(model_file);
}

Refiner::~Refiner()
{
  destroy_window();
}

// create window
void Refiner::create_window(int width, int height)
{
  pangolin::CreateWindowAndBind("Refiner", 640*2, 480*2);

  gtView_ = &pangolin::Display("gt").SetAspect(640.0/480.);
  poseView_ = &pangolin::Display("pose").SetAspect(640.0/480.);
  colorView_ = &pangolin::Display("color").SetAspect(640.0/480.);
  labelView_ = &pangolin::Display("label").SetAspect(640.0/480.);
  maskView_ = &pangolin::Display("mask").SetAspect(640.0/480.);

  multiView_ = &pangolin::Display("multi")
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(*gtView_)
      .AddDisplay(*colorView_)
      .AddDisplay(*labelView_)
      .AddDisplay(*poseView_)
      .AddDisplay(*maskView_);

  // create render
  renderer_ = new df::GLRenderer<ForegroundRenderType>(width, height);
}


void Refiner::destroy_window()
{
  pangolin::DestroyWindow("Refiner");
  delete renderer_;
}

// read the 3D models
void Refiner::loadModels(const std::string filename)
{
  std::ifstream stream(filename);
  std::vector<std::string> model_names;
  std::string name;

  while ( std::getline (stream, name) )
  {
    std::cout << name << std::endl;
    model_names.push_back(name);
  }

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
  texturedIndices_.resize(num_models);
  texturedCoords_.resize(num_models);
  texturedTextures_.resize(num_models);

  for (int m = 0; m < num_models; m++)
    initializeBuffers(assimpMeshes_[m], texture_names_[m], texturedVertices_[m], texturedIndices_[m], texturedCoords_[m], texturedTextures_[m], true);
}

aiMesh* Refiner::loadTexturedMesh(const std::string filename, std::string & texture_name)
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


void Refiner::initializeBuffers(aiMesh* assimpMesh, std::string textureName, 
  pangolin::GlBuffer & vertices, pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured)
{
    // std::cout << "loading vertices..." << std::endl;
    // std::cout << assimpMesh->mNumVertices << std::endl;
    vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    // std::cout << "loading normals..." << std::endl;
    std::vector<uint3> faces3(assimpMesh->mNumFaces);
    for (std::size_t i = 0; i < assimpMesh->mNumFaces; ++i) {
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


// feed data
void Refiner::feed_data(int width, int height, unsigned char* data, unsigned char* labels, pangolin::GlTexture & colorTex, pangolin::GlTexture & labelTex)
{
  // color image
  colorTex.Upload(data, GL_BGR, GL_UNSIGNED_BYTE);

  // process labels
  unsigned char* p = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
  for (int i = 0; i < width * height; i++)
  {
    int label = labels[i];
    p[3 * i] = class_colors[label][0];
    p[3 * i + 1] = class_colors[label][1];
    p[3 * i + 2] = class_colors[label][2];
  }

  labelTex.Upload(p, GL_RGB, GL_UNSIGNED_BYTE);
  free(p);
}


void Refiner::render(unsigned char* data, unsigned char* labels, float* rois, int num_rois, int num_gt, int width, int height, int num_classes,
                    float* poses_gt, float* poses_pred, float fx, float fy, float px, float py, float* extents, float* poses_new, int is_save)
{
  bool is_textured = true;

  pangolin::GlTexture colorTex(width, height);
  pangolin::GlTexture labelTex(width, height);

  // feed the first image in the batch only
  feed_data(width, height, data, labels, colorTex, labelTex);

  // show image
  colorView_->ActivateScissorAndClear();
  colorTex.RenderToViewportFlipY();

  // draw rois
  glDisable(GL_DEPTH_TEST);
  glColor3ub(0, 255, 0);
  for (int i = 0; i < num_rois; i++)
  {
    float x1 = 2 * rois[6 * i + 2] / width - 1;
    float y1 = -1 * (2 * rois[6 * i + 3] / height - 1);
    float x2 = 2 * rois[6 * i + 4] / width - 1;
    float y2 = -1 * (2 * rois[6 * i + 5] / height - 1);
    glBegin(GL_LINE_LOOP);
      glVertex2f(x1,y1);
      glVertex2f(x2,y1);
      glVertex2f(x2,y2);
      glVertex2f(x1,y2);
    glEnd();
  }

  // show label
  glColor3ub(255,255,255);
  labelView_->ActivateScissorAndClear();
  labelTex.RenderToViewportFlipY();

  // show gt pose
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();

  for (int n = 0; n < num_gt; n++)
  {
    pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, 0.25, 6.0);

    int class_id = int(poses_gt[n * 13 + 1]);

    Eigen::Quaterniond quaternion(poses_gt[n * 13 + 6], poses_gt[n * 13 + 7], poses_gt[n * 13 + 8], poses_gt[n * 13 + 9]);
    Sophus::SE3d::Point translation(poses_gt[n * 13 + 10], poses_gt[n * 13 + 11], poses_gt[n * 13 + 12]);
    const Sophus::SE3d T_co(quaternion, translation);

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = T_co.cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id-1].Bind();
    texturedVertices_[class_id-1].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id-1].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id-1].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id-1].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id-1].Unbind();
    texturedTextures_[class_id-1].Unbind();
    texturedVertices_[class_id-1].Unbind();
    texturedCoords_[class_id-1].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
  }

  // show predicted pose
  glColor3ub(255,255,255);
  poseView_->ActivateScissorAndClear();

  for (int n = 0; n < num_rois; n++)
  {
    pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, 0.25, 6.0);

    int class_id = int(rois[n * 6 + 1]);

    // find the gt index
    int gt_ind = -1;
    for (int i = 0; i < num_gt; i++)
    {
      int gt_id = int(poses_gt[i * 13 + 1]);
      if(class_id == gt_id)
      {
        gt_ind = i;
        break;
      }
    }

    if (gt_ind == -1)
    {
      std::cout << "detection " << n << " does not match any gt" << std::endl;
      continue;
    }

    Eigen::Quaterniond quaternion_pred(poses_pred[n * 7 + 0], poses_pred[n * 7 + 1], poses_pred[n * 7 + 2], poses_pred[n * 7 + 3]);
    Sophus::SE3d::Point translation_pred(poses_pred[n * 7 + 4], poses_pred[n * 7 + 5], poses_pred[n * 7 + 6]);
    const Sophus::SE3d T_co_pred(quaternion_pred, translation_pred);

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = T_co_pred.cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);

    mvMatrix.Load();

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id-1].Bind();
    texturedVertices_[class_id-1].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id-1].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id-1].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id-1].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id-1].Unbind();
    texturedTextures_[class_id-1].Unbind();
    texturedVertices_[class_id-1].Unbind();
    texturedCoords_[class_id-1].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
  }

  // refine the pose
  refine(labels, rois, num_rois, width, height, num_classes, poses_pred, fx, fy, px, py, extents, poses_new);

  // display the new poses
  glColor3ub(255,255,255);
  maskView_->ActivateScissorAndClear();

  for (int n = 0; n < num_rois; n++)
  {
    pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, 0.25, 6.0);

    int class_id = int(rois[n * 6 + 1]);

    // find the gt index
    int gt_ind = -1;
    for (int i = 0; i < num_gt; i++)
    {
      int gt_id = int(poses_gt[i * 13 + 1]);
      if(class_id == gt_id)
      {
        gt_ind = i;
        break;
      }
    }

    if (gt_ind == -1)
    {
      std::cout << "detection " << n << " does not match any gt" << std::endl;
      continue;
    }

    Eigen::Quaterniond quaternion_pred(poses_new[n * 7 + 0], poses_new[n * 7 + 1], poses_new[n * 7 + 2], poses_new[n * 7 + 3]);
    Sophus::SE3d::Point translation_pred(poses_new[n * 7 + 4], poses_new[n * 7 + 5], poses_new[n * 7 + 6]);
    const Sophus::SE3d T_co_pred(quaternion_pred, translation_pred);

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = T_co_pred.cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);

    mvMatrix.Load();

    glEnable(GL_TEXTURE_2D);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    texturedTextures_[class_id-1].Bind();
    texturedVertices_[class_id-1].Bind();
    glVertexPointer(3,GL_FLOAT,0,0);
    texturedCoords_[class_id-1].Bind();
    glTexCoordPointer(2,GL_FLOAT,0,0);
    texturedIndices_[class_id-1].Bind();
    glDrawElements(GL_TRIANGLES, texturedIndices_[class_id-1].num_elements, GL_UNSIGNED_INT, 0);
    texturedIndices_[class_id-1].Unbind();
    texturedTextures_[class_id-1].Unbind();
    texturedVertices_[class_id-1].Unbind();
    texturedCoords_[class_id-1].Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
  }

  if (is_save)
  {
    std::string filename = std::to_string(counter_++);
    pangolin::SaveWindowOnRender(filename);
  }
  pangolin::FinishFrame();
}


inline std::vector<cv::Point3f> Refiner::getBB3D(const cv::Vec<float, 3>& extent)
{
    std::vector<cv::Point3f> bb;
    
    float xHalf = extent[0] * 0.5;
    float yHalf = extent[1] * 0.5;
    float zHalf = extent[2] * 0.5;
    
    bb.push_back(cv::Point3f(xHalf, yHalf, zHalf));
    bb.push_back(cv::Point3f(-xHalf, yHalf, zHalf));
    bb.push_back(cv::Point3f(xHalf, -yHalf, zHalf));
    bb.push_back(cv::Point3f(-xHalf, -yHalf, zHalf));
    
    bb.push_back(cv::Point3f(xHalf, yHalf, -zHalf));
    bb.push_back(cv::Point3f(-xHalf, yHalf, -zHalf));
    bb.push_back(cv::Point3f(xHalf, -yHalf, -zHalf));
    bb.push_back(cv::Point3f(-xHalf, -yHalf, -zHalf));
    
    return bb;
}

// get 3D bounding boxes
void Refiner::getBb3Ds(float* extents, std::vector<std::vector<cv::Point3f>>& bb3Ds, int num_classes)
{
  // for each object
  for (int i = 1; i < num_classes; i++)
  {
    cv::Vec<float, 3> extent;
    extent(0) = extents[i * 3];
    extent(1) = extents[i * 3 + 1];
    extent(2) = extents[i * 3 + 2];

    bb3Ds.push_back(getBB3D(extent));
  }
}


void Refiner::refine(unsigned char* labels, float* rois, int num_rois, int width, int height, int num_classes,
                    float* poses_pred, float fx, float fy, float px, float py, float* extents, float* poses_new)
{
  int poseIterations = 100;

  // bb3Ds
  std::vector<std::vector<cv::Point3f>> bb3Ds;
  getBb3Ds(extents, bb3Ds, num_classes);

  // camera matrix
  cv::Mat_<float> camMat = cv::Mat_<float>::zeros(3, 3);
  camMat(0, 0) = fx;
  camMat(1, 1) = fy;
  camMat(2, 2) = 1.f;
  camMat(0, 2) = px;
  camMat(1, 2) = py;

  // create the masks of predicted labels
  std::vector<cv::Mat*> masks(num_classes);
  for (int i = 0; i < num_classes; i++)
    masks[i] = new cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));

  for(int x = 0; x < width; x++)
  {
    for(int y = 0; y < height; y++)
    {
      int label = labels[y * width + x];
      masks[label]->at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
    }
  }

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), 0.25, 6.0);

  // for each roi
  for (int i = 0; i < num_rois; i++)
  {
    int class_id = int(rois[i * 6 + 1]);

    // 2D bounding box
    cv::Rect bb2D(rois[i * 6 + 2], rois[i * 6 + 3], rois[i * 6 + 4] - rois[i * 6 + 2], rois[i * 6 + 5] - rois[i * 6 + 3]);

    // 3D bounding box
    std::vector<cv::Point3f> bb3D = bb3Ds[class_id-1];

    // construct the data
    DataForOpt data;
    data.width = width;
    data.height = height;
    data.bb2D = bb2D;
    data.bb3D = bb3D;
    data.camMat = camMat;
    data.projectionMatrix = projectionMatrix;
    data.texturedVertices = &texturedVertices_[class_id-1];
    data.texturedIndices = &texturedIndices_[class_id-1];
    data.gt_mask = masks[class_id];
    data.view = maskView_;
    data.renderer = renderer_;

    // initialize pose
    std::vector<double> vec(7);
    vec[0] = poses_pred[i * 7 + 0];
    vec[1] = poses_pred[i * 7 + 1];
    vec[2] = poses_pred[i * 7 + 2];
    vec[3] = poses_pred[i * 7 + 3];
    vec[4] = poses_pred[i * 7 + 4];
    vec[5] = poses_pred[i * 7 + 5];
    vec[6] = poses_pred[i * 7 + 6];

    printf("before\n");
    for (int j = 0; j < 7; j++)
      printf("%.2f ", vec[j]);
    printf("\n");

    // optimization
    poseWithOpt(vec, data, poseIterations);

    printf("after\n");
    for (int j = 0; j < 7; j++)
      printf("%.2f ", vec[j]);
    printf("\n");

    poses_new[i * 7 + 0] = vec[0];
    poses_new[i * 7 + 1] = vec[1];
    poses_new[i * 7 + 2] = vec[2];
    poses_new[i * 7 + 3] = vec[3];
    poses_new[i * 7 + 4] = vec[4];
    poses_new[i * 7 + 5] = vec[5];
    poses_new[i * 7 + 6] = vec[6];

  }
  
  // release the masks
  for (int i = 0; i < num_classes; i++)
    delete masks[i];
}


inline cv::Rect getBB2D(int imageWidth, int imageHeight, const std::vector<cv::Point3f>& bb3D, const cv::Mat& camMat, const cv::Mat& RT)
{   
    // project 3D bounding box vertices into the image
    std::vector<cv::Point2f> bb2D;

    cv::Mat P = camMat * RT;

    // projection
    for (int i = 0; i < bb3D.size(); i++)
    {
      cv::Mat x3d = cv::Mat::zeros(4, 1, CV_32F);
      x3d.at<float>(0, 0) = bb3D[i].x;
      x3d.at<float>(1, 0) = bb3D[i].y;
      x3d.at<float>(2, 0) = bb3D[i].z;
      x3d.at<float>(3, 0) = 1.0;

      cv::Mat x2d = P * x3d;
      bb2D.push_back(cv::Point2f(x2d.at<float>(0, 0) / x2d.at<float>(2, 0), x2d.at<float>(1, 0) / x2d.at<float>(2, 0)));
    }
    
    // get min-max of projected vertices
    int minX = imageWidth - 1;
    int maxX = 0;
    int minY = imageHeight - 1;
    int maxY = 0;
    
    for(unsigned j = 0; j < bb2D.size(); j++)
    {
	minX = std::min((float) minX, bb2D[j].x);
	minY = std::min((float) minY, bb2D[j].y);
	maxX = std::max((float) maxX, bb2D[j].x);
	maxY = std::max((float) maxY, bb2D[j].y);
    }
    
    // clamp at image border
    minX = clamp(minX, 0, imageWidth - 1);
    maxX = clamp(maxX, 0, imageWidth - 1);
    minY = clamp(minY, 0, imageHeight - 1);
    maxY = clamp(maxY, 0, imageHeight - 1);
    
    return cv::Rect(minX, minY, (maxX - minX + 1), (maxY - minY + 1));
}


int clamp(int val, int min_val, int max_val)
{
    return std::max(min_val, std::min(max_val, val));
}

inline float getIoU(const cv::Rect& bb1, const cv::Rect bb2)
{
    cv::Rect intersection = bb1 & bb2;
    return (intersection.area() / (float) (bb1.area() + bb2.area() - intersection.area()));
}

static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
{
  DataForOpt* dataForOpt = (DataForOpt*) data;

  // model view matrix
  Eigen::Quaterniond quaternion(pose[0], pose[1], pose[2], pose[3]);
  Sophus::SE3d::Point translation(pose[4], pose[5], pose[6]);
  const Sophus::SE3d T_co(quaternion, translation);
  const Eigen::Matrix<double, 3, 4> RT = T_co.matrix3x4();

  // convert RT to opencv matrix
  cv::Mat_<float> trans = cv::Mat_<float>::zeros(3, 4);
  for (int i = 0; i < 3; i++)
  {
    for(int j = 0; j < 4; j++)
      trans(i, j) = RT(i, j);
  }

  // project the 3D bounding box according to the current pose
  cv::Rect bb2D = getBB2D(dataForOpt->width, dataForOpt->height, dataForOpt->bb3D, dataForOpt->camMat, trans);

  // compute IoU between boxes
  float IoU_box = getIoU(bb2D, dataForOpt->bb2D);

  // project the 3D model
  dataForOpt->renderer->setProjectionMatrix(dataForOpt->projectionMatrix);
  dataForOpt->renderer->setModelViewMatrix(T_co.cast<float>().matrix());
  dataForOpt->renderer->render( { dataForOpt->texturedVertices }, *(dataForOpt->texturedIndices) );
  glColor3ub(255,255,255);
  dataForOpt->view->ActivateScissorAndClear();
  dataForOpt->renderer->texture(0).RenderToViewportFlipY();
        
  // download the mask
  cv::Mat mask(dataForOpt->height, dataForOpt->width, CV_8UC3);
  dataForOpt->renderer->texture(0).Download(mask.data, GL_RGB, GL_UNSIGNED_BYTE);

  // compute the overlap between masks
  cv::Mat dst1, dst2;
  cv::bitwise_and(mask, *(dataForOpt->gt_mask), dst1);
  cv::bitwise_or(mask, *(dataForOpt->gt_mask), dst2);
  float IoU_seg = cv::sum(dst1)[0] / cv::sum(dst2)[0];

  // compute IoU between boxes
  float energy = -1 * (0.5 * IoU_box + IoU_seg);

  return energy;
}


double poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations) 
{
  // set up optimization algorithm (gradient free)
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 7); 

  // set optimization bounds 
  double rotRange = 0.2;
  double tRangeXY = 0.1;
  double tRangeZ = 0.2; // pose uncertainty is larger in Z direction
	
  std::vector<double> lb(7);
  lb[0] = vec[0]-rotRange;
  lb[1] = vec[1]-rotRange;
  lb[2] = vec[2]-rotRange;
  lb[3] = vec[3]-rotRange;
  lb[4] = vec[4]-tRangeXY;
  lb[5] = vec[5]-tRangeXY;
  lb[6] = vec[6]-tRangeZ;
  opt.set_lower_bounds(lb);
      
  std::vector<double> ub(7);
  ub[0] = vec[0]+rotRange;
  ub[1] = vec[1]+rotRange;
  ub[2] = vec[2]+rotRange;
  ub[3] = vec[3]+rotRange;
  ub[4] = vec[4]+tRangeXY;
  ub[5] = vec[5]+tRangeXY;
  ub[6] = vec[6]+tRangeZ;
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


int main(int argc, char** argv) 
{
  Refiner refiner;
  refiner.setup(argv[1]);

  //while (!pangolin::ShouldQuit()) 
  refiner.render(NULL, NULL, NULL, 0, 0, 480, 640, 22, NULL, NULL, 0, 0, 0, 0, NULL, NULL, 0);
}
