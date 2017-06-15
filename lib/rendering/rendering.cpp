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

  multiView_ = &pangolin::Display("multi")
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
  texture_names_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    assimpMeshes_[m] = loadTexturedMesh(model_names[m], texture_names_[m]);
    std::cout << texture_names_[m] << std::endl;
  }
}

aiMesh* Render::loadTexturedMesh(const std::string filename, std::string & texture_name)
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


void Render::initializeBuffers(aiMesh* assimpMesh, std::string textureName, 
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


float Render::render(const float* data, const int* labels, const float* rois, int num_rois, int num_gt, int num_classes,
                    const float* poses_gt, const float* poses_pred, const float* poses_init, float* bottom_diff)
{
  bool is_textured = false;
  create_window();

  // initialize buffers
  pangolin::GlBuffer texturedVertices_;
  pangolin::GlBuffer texturedIndices_;
  pangolin::GlBuffer texturedCoords_;
  pangolin::GlTexture texturedTextures_;

  pangolin::GlTexture colorTex(color_camera_->width(), color_camera_->height());
  pangolin::GlTexture labelTex(color_camera_->width(), color_camera_->height());
  feed_data(data, labels, colorTex, labelTex);

  // show image
  colorView_->ActivateScissorAndClear();
  colorTex.RenderToViewportFlipY();

  // draw rois
  glColor3ub(0, 255, 0);
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
  if (is_textured)
  {
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glColor3ub(255,255,255);
    gtView_->ActivateScissorAndClear();
    pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(color_camera_->width(), color_camera_->height(),
                                                                                        color_camera_->params()[0], color_camera_->params()[1],
                                                                                        color_camera_->params()[2]+0.5,
                                                                                        color_camera_->params()[3]+0.5,
                                                                                        0.25,6.0);

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);
  }

  std::vector<cv::Mat*> gt_masks(num_gt);
  for (int n = 0; n < num_gt; n++)
  {
    int class_id = int(poses_gt[n * 13 + 1]);

    initializeBuffers(assimpMeshes_[class_id-1], texture_names_[class_id-1], texturedVertices_, texturedIndices_, texturedCoords_, texturedTextures_, is_textured);
    Eigen::Quaterniond quaternion(poses_gt[n * 13 + 6], poses_gt[n * 13 + 7], poses_gt[n * 13 + 8], poses_gt[n * 13 + 9]);
    Sophus::SE3d::Point translation(poses_gt[n * 13 + 10], poses_gt[n * 13 + 11], poses_gt[n * 13 + 12]);
    const Sophus::SE3d T_co(quaternion, translation);

    if (is_textured)
    {
      Eigen::Matrix4f mv = T_co.cast<float>().matrix();
      pangolin::OpenGlMatrix mvMatrix(mv);

      mvMatrix.Load();

      glEnable(GL_TEXTURE_2D);
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      texturedTextures_.Bind();
      texturedVertices_.Bind();
      glVertexPointer(3,GL_FLOAT,0,0);
      texturedCoords_.Bind();
      glTexCoordPointer(2,GL_FLOAT,0,0);
      texturedIndices_.Bind();
      glDrawElements(GL_TRIANGLES, texturedIndices_.num_elements, GL_UNSIGNED_INT, 0);
      texturedIndices_.Unbind();
      texturedTextures_.Unbind();
      texturedVertices_.Unbind();
      texturedCoords_.Unbind();
      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
      glDisable(GL_TEXTURE_2D);
    }
    else
    {
      renderer_->setModelViewMatrix(T_co.cast<float>().matrix());
      renderer_->render( { &texturedVertices_ }, texturedIndices_ );
      glColor3ub(255,255,255);
      gtView_->ActivateScissorAndClear();
      renderer_->texture(0).RenderToViewportFlipY();

      // download the mask
      gt_masks[n] = new cv::Mat(color_camera_->height(), color_camera_->width(), CV_8UC3);
      renderer_->texture(0).Download(gt_masks[n]->data, GL_RGB, GL_UNSIGNED_BYTE);
    }
  }

  // show predicted pose
  if (is_textured)
  {
    glColor3ub(255,255,255);
    poseView_->ActivateScissorAndClear();
    pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(color_camera_->width(), color_camera_->height(),
                                                                                        color_camera_->params()[0], color_camera_->params()[1],
                                                                                        color_camera_->params()[2]+0.5,
                                                                                        color_camera_->params()[3]+0.5,
                                                                                        0.25,6.0);

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);
  }

  cv::Mat dst1;
  cv::Mat dst2;
  double delta = 0.001;
  float loss = 0;
  for (int n = 0; n < num_rois; n++)
  {
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

    initializeBuffers(assimpMeshes_[class_id-1], texture_names_[class_id-1], texturedVertices_, texturedIndices_, texturedCoords_, texturedTextures_, is_textured);

    // render mulitple times
    int num = 5;
    std::vector<double> IoUs(num);
    for (int i = 0; i < num; i++)
    {
      double w, x, y, z;
      if (i == 0)
      {
        w = poses_pred[n * 4 * num_classes + 4 * class_id + 0];
        x = poses_pred[n * 4 * num_classes + 4 * class_id + 1];
        y = poses_pred[n * 4 * num_classes + 4 * class_id + 2];
        z = poses_pred[n * 4 * num_classes + 4 * class_id + 3];
      }
      else if(i == 1)
      {
        w = poses_pred[n * 4 * num_classes + 4 * class_id + 0] + delta;
        x = poses_pred[n * 4 * num_classes + 4 * class_id + 1];
        y = poses_pred[n * 4 * num_classes + 4 * class_id + 2];
        z = poses_pred[n * 4 * num_classes + 4 * class_id + 3];
      }
      else if(i == 2)
      {
        w = poses_pred[n * 4 * num_classes + 4 * class_id + 0];
        x = poses_pred[n * 4 * num_classes + 4 * class_id + 1] + delta;
        y = poses_pred[n * 4 * num_classes + 4 * class_id + 2];
        z = poses_pred[n * 4 * num_classes + 4 * class_id + 3];
      }
      else if(i == 3)
      {
        w = poses_pred[n * 4 * num_classes + 4 * class_id + 0];
        x = poses_pred[n * 4 * num_classes + 4 * class_id + 1];
        y = poses_pred[n * 4 * num_classes + 4 * class_id + 2] + delta;
        z = poses_pred[n * 4 * num_classes + 4 * class_id + 3];
      }
      else
      {
        w = poses_pred[n * 4 * num_classes + 4 * class_id + 0];
        x = poses_pred[n * 4 * num_classes + 4 * class_id + 1];
        y = poses_pred[n * 4 * num_classes + 4 * class_id + 2];
        z = poses_pred[n * 4 * num_classes + 4 * class_id + 3] + delta;
      }

      Eigen::Quaterniond quaternion_pred(w, x, y, z);
      Sophus::SE3d::Point translation_pred(poses_init[n * 7 + 4], poses_init[n * 7 + 5], poses_init[n * 7 + 6]);
      const Sophus::SE3d T_co_pred(quaternion_pred, translation_pred);

      if (is_textured)
      {
        Eigen::Matrix4f mv = T_co_pred.cast<float>().matrix();
        pangolin::OpenGlMatrix mvMatrix(mv);

        mvMatrix.Load();

        glEnable(GL_TEXTURE_2D);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        texturedTextures_.Bind();
        texturedVertices_.Bind();
        glVertexPointer(3,GL_FLOAT,0,0);
        texturedCoords_.Bind();
        glTexCoordPointer(2,GL_FLOAT,0,0);
        texturedIndices_.Bind();
        glDrawElements(GL_TRIANGLES, texturedIndices_.num_elements, GL_UNSIGNED_INT, 0);
        texturedIndices_.Unbind();
        texturedTextures_.Unbind();
        texturedVertices_.Unbind();
        texturedCoords_.Unbind();
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisable(GL_TEXTURE_2D);
      }
      else
      {
        renderer_->setModelViewMatrix(T_co_pred.cast<float>().matrix());
        renderer_->render( { &texturedVertices_ }, texturedIndices_ );
        poseView_->ActivateScissorAndClear();
        renderer_->texture(0).RenderToViewportFlipY();

        // download the mask
        cv::Mat mask(color_camera_->height(), color_camera_->width(), CV_8UC3);
        renderer_->texture(0).Download(mask.data, GL_RGB, GL_UNSIGNED_BYTE);

        // compute the overlap between masks
        cv::bitwise_and(mask, *gt_masks[gt_ind], dst1);
        cv::bitwise_or(mask, *gt_masks[gt_ind], dst2);
        IoUs[i] = cv::sum(dst1)[0] / cv::sum(dst2)[0];
      }
    }  // end rendering

    // compute loss and gradient
    loss += (1.0 - IoUs[0]) / num_rois;

    bottom_diff[n * 4 * num_classes + 4 * class_id + 0] = (IoUs[0] - IoUs[1]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 1] = (IoUs[0] - IoUs[2]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 2] = (IoUs[0] - IoUs[3]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 3] = (IoUs[0] - IoUs[4]) / delta / num_rois;
  }

  // pangolin::SaveWindowOnRender("window");
  pangolin::FinishFrame();

  // usleep( 2000 * 1000 );
  destroy_window();

  for (int n = 0; n < num_gt; n++)
    delete gt_masks[n];

  return loss;
}

int main(int argc, char** argv) 
{
  Render render;
  render.setup(argv[1], argv[2]);

  //while (!pangolin::ShouldQuit()) 
  render.render(NULL, NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL);
}
