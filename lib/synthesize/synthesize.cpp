#include "synthesize.hpp"
#include "thread_rand.h"

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
  counter_ = 0;
  create_window(640.0, 480.0);

  loadModels(model_file);
  std::cout << "loaded models" << std::endl;

  loadPoses(pose_file);
  std::cout << "loaded poses" << std::endl;
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
              unsigned char* color, float* depth, float* vertmap, float* class_indexes, float *poses_return)
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

  // sample object classes
  std::vector<int> class_ids(num);
  for (int i = 0; i < num; )
  {
    int class_id = irand(0, pose_nums_.size());
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
  }

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


int main(int argc, char** argv) 
{
  Synthesizer Synthesizer(argv[1], argv[2]);

  // camera parameters
  int width = 640;
  int height = 480;
  float fx = 1066.778, fy = 1067.487, px = 312.9869, py = 241.3109, zfar = 6.0, znear = 0.25;

  while (!pangolin::ShouldQuit()) 
  {
    clock_t start = clock();    

    Synthesizer.render(width, height, fx, fy, px, py, znear, zfar, NULL, NULL, NULL, NULL, NULL);

    clock_t stop = clock();    
    double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Time elapsed in ms: %f\n", elapsed);
  }
}
