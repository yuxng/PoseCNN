#include "rendering.hpp"

using namespace df;

Render::Render(std::string rig_specification_file, std::string model_file)
{
  setup_cameras(rig_specification_file);
  loadModels(model_file);
}


void Render::setup(std::string rig_specification_file, std::string model_file)
{
  setup_cameras(rig_specification_file);
  loadModels(model_file);
}

// setup cameras
void Render::setup_cameras(std::string rig_specification_file)
{
  int color_stream_index = 0;
  int depth_stream_index = 1;

  // camera configuration
  std::ifstream rig_stream(rig_specification_file);
  picojson::value val;
  rig_stream >> val;
  if (!val.contains("rig")) 
    throw std::runtime_error("could not find rig");

  picojson::value rig_val = val["rig"];
  if (!rig_val.contains("camera")) 
    throw std::runtime_error("could not find camera");

  rig_ = new Rig<double>(rig_val);
  if (rig_->numCameras() != 2)
    throw std::runtime_error("expected a rig configuration with 2 cameras (RGB + depth)");

  color_camera_ = &rig_->camera(color_stream_index);
  depth_camera_ = &rig_->camera(depth_stream_index);
}

// create window
void Render::create_window()
{
  pangolin::CreateWindowAndBind("Render", 640, 480);

  gtView_ = &pangolin::Display("gt").SetAspect(640.0/480.);
  poseView_ = &pangolin::Display("pose").SetAspect(640.0/480.);
  colorView_ = &pangolin::Display("color").SetAspect(640.0/480.);
  labelView_ = &pangolin::Display("label").SetAspect(640.0/480.);

  pangolin::Display("multi")
      .SetLayout(pangolin::LayoutEqual)
      .AddDisplay(*gtView_)
      .AddDisplay(*poseView_)
      .AddDisplay(*colorView_)
      .AddDisplay(*labelView_);

  // create render
  renderer_ = new df::GLRenderer<ForegroundRenderType>(color_camera_->width(), color_camera_->height());
  rendererCam_ = new pangolin::OpenGlRenderState(pangolin::ProjectionMatrixRDF_TopLeft(color_camera_->width(), color_camera_->height(),
                                                                                  color_camera_->params()[0], -color_camera_->params()[1],
                                                                                  color_camera_->params()[2]+0.5,
                                                                                  color_camera_->height()-(color_camera_->params()[3]+0.5),
                                                                                  0.25,6.0));
  renderer_->setProjectionMatrix(rendererCam_->GetProjectionMatrix());
}


void Render::destroy_window()
{
  pangolin::DestroyWindow("Render");
  delete renderer_;
  delete rendererCam_;
}

// read the 3D models
void Render::loadModels(const std::string filename)
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

  for (int m = 0; m < num_models; ++m)
    assimpMeshes_[m] = loadTexturedMesh(model_names[m]);
}

aiMesh* Render::loadTexturedMesh(const std::string filename)
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

    return assimpMesh;
}


void Render::initializeBuffers(aiMesh* assimpMesh, pangolin::GlBuffer & vertices, pangolin::GlBuffer & indices)
{
    std::cout << "loading vertices..." << std::endl;
    std::cout << assimpMesh->mNumVertices << std::endl;
    vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    std::cout << "loading normals..." << std::endl;
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
}


// feed data
void Render::feed_data(const float* data, const int* labels, pangolin::GlTexture & colorTex, pangolin::GlTexture & labelTex)
{
  int width = color_camera_->width();
  int height = color_camera_->height();

  // convert depth values
  unsigned char* p = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);

  for (int i = 0; i < width * height * 3; i++)
  {
    switch(i % 3)
    {
      case 0:
        p[i] = (unsigned char)(data[i] + 102.9801);
        break;
      case 1:
        p[i] = (unsigned char)(data[i] + 115.9465);
        break;
      case 2:
        p[i] = (unsigned char)(data[i] + 122.7717);
        break;
    }
  }

  colorTex.Upload(p, GL_BGR, GL_UNSIGNED_BYTE);
  free(p);

  // process labels
  p = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
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


void Render::render(const float* data, const int* labels, const float* rois, int num_rois)
{
  create_window();

  // initialize buffers
  pangolin::GlBuffer texturedVertices_;
  pangolin::GlBuffer texturedIndices_;
  initializeBuffers(assimpMeshes_[0], texturedVertices_, texturedIndices_);

  pangolin::GlTexture colorTex(color_camera_->width(), color_camera_->height());
  pangolin::GlTexture labelTex(color_camera_->width(), color_camera_->height());
  feed_data(data, labels, colorTex, labelTex);

  // show image
  colorView_->ActivateScissorAndClear();
  colorTex.RenderToViewportFlipY();
  glColor3ub(0, 255, 0);

  // draw rois
  int width = color_camera_->width();
  int height = color_camera_->height();
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
  Eigen::Quaterniond quaternion(5.43034673e-01, 8.05942714e-01, 1.70454860e-01,-1.62833780e-01);
  Sophus::SE3d::Point translation(-9.21968296e-02, -6.99419007e-02, 9.75155234e-01);

  const Sophus::SE3d T_co(quaternion, translation);
  std::cout << T_co.matrix3x4() << std::endl;

  renderer_->setModelViewMatrix(T_co.cast<float>().matrix());
  renderer_->render( { &texturedVertices_ }, texturedIndices_ );

  glColor3ub(255,255,255);
  glClearColor(1,0,0,1);
  gtView_->ActivateScissorAndClear();
  renderer_->texture(0).RenderToViewportFlipY();

  // df::ManagedHostTensor2<Eigen::Matrix<char,3,1> > objectMask({image_width_, image_height_});
  // renderer_->texture(0).Download(objectMask.data(), GL_RGB, GL_UNSIGNED_BYTE);


  glColor3ub(255,255,255);
  // glClearColor(0,1,0,1);
  poseView_->ActivateScissorAndClear();

  pangolin::FinishFrame();

  usleep( 2000 * 1000 );
  destroy_window();
}

int main(int argc, char** argv) 
{
  Render render;
  render.setup(argv[1], argv[2]);

  //while (!pangolin::ShouldQuit()) 
  render.render(NULL, NULL, NULL, 0);
}
