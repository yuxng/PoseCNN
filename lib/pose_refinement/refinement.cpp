#include "refinement.hpp"

using namespace df;

Refiner::Refiner(std::string model_file)
{
  counter_ = 0;
  create_window(640, 480);

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
  pangolin::CreateWindowAndBind("Refiner", 640, 480);

  gtView_ = &pangolin::Display("gt").SetAspect(640.0/480.);
  poseView_ = &pangolin::Display("pose").SetAspect(640.0/480.);
  colorView_ = &pangolin::Display("color").SetAspect(640.0/480.);
  labelView_ = &pangolin::Display("label").SetAspect(640.0/480.);

  multiView_ = &pangolin::Display("multi")
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(*gtView_)
      .AddDisplay(*poseView_)
      .AddDisplay(*colorView_)
      .AddDisplay(*labelView_);
}


void Refiner::destroy_window()
{
  pangolin::DestroyWindow("Refiner");
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
  labelTex.Upload(labels, GL_RGB, GL_UNSIGNED_BYTE);
}


void Refiner::render(unsigned char* data, unsigned char* labels, float* rois, int num_rois, int num_gt, int width, int height,
                    float* poses_gt, float* poses_pred, float fx, float fy, float px, float py)
{
  bool is_textured = true;

  // initialize buffers
  // pangolin::GlBuffer texturedVertices_;
  // pangolin::GlBuffer texturedIndices_;
  // pangolin::GlBuffer texturedCoords_;
  // pangolin::GlTexture texturedTextures_;

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

    // initializeBuffers(assimpMeshes_[class_id-1], texture_names_[class_id-1], texturedVertices_, texturedIndices_, texturedCoords_, texturedTextures_, is_textured);
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

    // initializeBuffers(assimpMeshes_[class_id-1], texture_names_[class_id-1], texturedVertices_, texturedIndices_, texturedCoords_, texturedTextures_, is_textured);

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

  std::string filename = std::to_string(counter_++);
  // pangolin::SaveWindowOnRender(filename);
  pangolin::FinishFrame();
}

int main(int argc, char** argv) 
{
  Refiner refiner;
  refiner.setup(argv[1]);

  //while (!pangolin::ShouldQuit()) 
  refiner.render(NULL, NULL, NULL, 0, 0, 480, 640, NULL, NULL, 0, 0, 0, 0);
}
