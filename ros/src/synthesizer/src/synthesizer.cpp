#include "synthesizer/synthesizer.hpp"

using namespace df;
static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data);

Synthesizer::Synthesizer(std::string model_file, std::string pose_file)
{
  model_file_ = model_file;
  pose_file_ = pose_file;
  counter_ = 0;
  setup_ = 0;
}

void Synthesizer::setup(int width, int height)
{
  create_window(width, height);

  loadModels(model_file_);
  std::cout << "loaded models" << std::endl;

  loadPoses(pose_file_);
  std::cout << "loaded poses" << std::endl;

  // create tensors

  // labels
  labels_device_ = new df::ManagedDeviceTensor2<int>({width, height});

  // depth map
  depth_map_ = new ManagedTensor<2, float>({width, height});
  depth_map_device_ = new ManagedTensor<2, float, DeviceResident>(depth_map_->dimensions());
  depth_factor_ = 1000.0;
  depth_cutoff_ = 20.0;

  // 3D points
  vertex_map_device_ = new ManagedDeviceTensor2<Vec3>({width, height});
  vertex_map_ = new ManagedHostTensor2<Vec3>({width, height});
  predicted_verts_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > ({width, height});
  predicted_normals_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > ({width, height});
  predicted_verts_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({width, height});
  predicted_normals_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({width, height});

  setup_ = 1;
}

Synthesizer::~Synthesizer()
{
  destroy_window();
}

// create window
void Synthesizer::create_window(int width, int height)
{
  pangolin::CreateWindowAndBind("Synthesizer", width, height);

  gtView_ = &pangolin::Display("gt").SetAspect(float(width)/float(height));

  // create render
  renderer_ = new df::GLRenderer<df::CanonicalVertRenderType>(width, height);
  renderer_vn_ = new df::GLRenderer<df::VertAndNormalRenderType>(width, height);
}


void Synthesizer::destroy_window()
{
  pangolin::DestroyWindow("Synthesizer");
  delete renderer_;
  delete renderer_vn_;
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
  std::vector<std::string> texture_names;
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
  texture_names.resize(num_models);
  std::cout << num_models << " models" << std::endl;

  for (int m = 0; m < num_models; ++m)
  {
    assimpMeshes_[m] = loadTexturedMesh(model_names[m], texture_names[m]);
    std::cout << texture_names[m] << std::endl;
  }

  // buffers
  texturedVertices_.resize(num_models);
  canonicalVertices_.resize(num_models);
  vertexColors_.resize(num_models);
  vertexNormals_.resize(num_models);
  texturedIndices_.resize(num_models);
  texturedCoords_.resize(num_models);
  texturedTextures_.resize(num_models);
  is_textured_.resize(num_models);

  for (int m = 0; m < num_models; m++)
  {
    bool is_textured;
    if (texture_names[m] == "")
      is_textured = false;
    else
      is_textured = true;
    is_textured_[m] = is_textured;

    initializeBuffers(m, assimpMeshes_[m], texture_names[m], texturedVertices_[m], canonicalVertices_[m], vertexColors_[m], vertexNormals_[m],
                      texturedIndices_[m], texturedCoords_[m], texturedTextures_[m], is_textured);
  }
}

aiMesh* Synthesizer::loadTexturedMesh(const std::string filename, std::string & texture_name)
{
    const struct aiScene * scene = aiImportFile(filename.c_str(), aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals);
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
    for (int i = 0; i < scene->mNumMaterials; ++i) 
    {
        aiMaterial * material = scene->mMaterials[i];
        std::cout << "diffuse: " << material->GetTextureCount(aiTextureType_DIFFUSE) << std::endl;
        std::cout << "specular: " << material->GetTextureCount(aiTextureType_SPECULAR) << std::endl;
        std::cout << "ambient: " << material->GetTextureCount(aiTextureType_AMBIENT) << std::endl;
        std::cout << "shininess: " << material->GetTextureCount(aiTextureType_SHININESS) << std::endl;

        if (material->GetTextureCount(aiTextureType_DIFFUSE)) 
        {
            aiString path;
            material->GetTexture(aiTextureType_DIFFUSE,0,&path);
            textureName = textureName + std::string(path.C_Str());
        }
    }

    aiMesh * assimpMesh = scene->mMeshes[0];
    std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
    std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;

    if (!assimpMesh->HasTextureCoords(0))
      texture_name = "";
    else
      texture_name = textureName;

    return assimpMesh;
}


void Synthesizer::initializeBuffers(int model_index, aiMesh* assimpMesh, std::string textureName,
  pangolin::GlBuffer & vertices, pangolin::GlBuffer & canonicalVertices, pangolin::GlBuffer & colors, pangolin::GlBuffer & normals,
  pangolin::GlBuffer & indices, pangolin::GlBuffer & texCoords, pangolin::GlTexture & texture, bool is_textured)
{
    std::cout << "number of vertices: " << assimpMesh->mNumVertices << std::endl;
    std::cout << "number of faces: " << assimpMesh->mNumFaces << std::endl;
    vertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    vertices.Upload(assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    // normals
    if (assimpMesh->HasNormals())
    {
      normals.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
      normals.Upload(assimpMesh->mNormals, assimpMesh->mNumVertices*sizeof(float)*3);
    }
    else
    {
      throw std::runtime_error("no normals in the mesh");
    }

    // canonical vertices
    std::vector<float3> canonicalVerts(assimpMesh->mNumVertices);
    std::memcpy(canonicalVerts.data(), assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float3));

    for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++)
      canonicalVerts[i].x += model_index;

    canonicalVertices.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
    canonicalVertices.Upload(canonicalVerts.data(), assimpMesh->mNumVertices*sizeof(float3));

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
    else
    {
      // vertex colors
      std::vector<float3> colors3(assimpMesh->mNumVertices);
      for (std::size_t i = 0; i < assimpMesh->mNumVertices; i++) 
      {
          aiColor4D & color = assimpMesh->mColors[0][i];
          colors3[i] = make_float3(color.r, color.g, color.b);
      }
      colors.Reinitialise(pangolin::GlArrayBuffer, assimpMesh->mNumVertices, GL_FLOAT, 3, GL_STATIC_DRAW);
      colors.Upload(colors3.data(), assimpMesh->mNumVertices*sizeof(float)*3);
    }
}


void Synthesizer::refinePose(int width, int height, int objID, float znear, float zfar,
  const int* labelmap, DataForOpt data, df::Poly3CameraModel<float> model, Sophus::SE3f & T_co, int iterations, float maxError, int algorithm)
{
  std::vector<pangolin::GlBuffer *> attributeBuffers({&texturedVertices_[objID - 1], &vertexNormals_[objID - 1]});
  renderer_vn_->setModelViewMatrix(T_co.matrix().cast<float>());
  renderer_vn_->render(attributeBuffers, texturedIndices_[objID - 1], GL_TRIANGLES);

  const pangolin::GlTextureCudaArray & vertTex = renderer_vn_->texture(0);
  const pangolin::GlTextureCudaArray & normTex = renderer_vn_->texture(1);

  // copy predicted normals
  {
    pangolin::CudaScopedMappedArray scopedArray(normTex);
    cudaMemcpy2DFromArray(predicted_normals_device_->data(), normTex.width*4*sizeof(float), *scopedArray, 0, 0, normTex.width*4*sizeof(float), normTex.height, cudaMemcpyDeviceToDevice);
    predicted_normals_->copyFrom(*predicted_normals_device_);
  }

  // copy predicted vertices
  {
    pangolin::CudaScopedMappedArray scopedArray(vertTex);
    cudaMemcpy2DFromArray(predicted_verts_device_->data(), vertTex.width*4*sizeof(float), *scopedArray, 0, 0, vertTex.width*4*sizeof(float), vertTex.height, cudaMemcpyDeviceToDevice);
    predicted_verts_->copyFrom(*predicted_verts_device_);
  }

  glColor3f(1, 1, 1);
  gtView_->ActivateScissorAndClear();
  renderer_vn_->texture(0).RenderToViewportFlipY();
  // pangolin::FinishFrame();

  switch (algorithm)
  {
    case 0:
    {
      // initialize pose
      std::vector<double> vec(7);
      vec[0] = 1;
      vec[1] = 0;
      vec[2] = 0;
      vec[3] = 0;
      vec[4] = 0;
      vec[5] = 0;
      vec[6] = 0;

      // optimization 
      float energy = poseWithOpt(vec, data, iterations);
      Eigen::Quaternionf quaternion(vec[0], vec[1], vec[2], vec[3]);
      Sophus::SE3f::Point translation(vec[4], vec[5], vec[6]);
      Sophus::SE3f update(quaternion, translation);
      T_co = update * T_co;
      break;     
    }
    case 1:
    {
      Eigen::Vector2f depthRange(znear, zfar);
      Sophus::SE3f update = icp(*vertex_map_device_, *predicted_verts_device_, *predicted_normals_device_,
                              model, T_co, depthRange, maxError, iterations);
      T_co = update * T_co;
      break;
    }
  }
}


// ICP
void Synthesizer::refineDistance(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, 
  float znear, float zfar, float factor, int num_roi, int channel_roi, const float* rois, const float* poses, 
  float* outputs, float* outputs_icp, std::vector<std::vector<geometry_msgs::Point32> >& output_points, float maxError)
{
  int iterations;
  if (setup_ == 0)
    setup(width, height);

  // build the camera paramters
  Eigen::Matrix<float,7,1,Eigen::DontAlign> params;
  params[0] = fx;
  params[1] = fy;
  params[2] = px;
  params[3] = py;
  params[4] = 0;
  params[5] = 0;
  params[6] = 0;
  df::Poly3CameraModel<float> model(params);

  // convert depth values
  float* p = depth_map_->data();
  ushort* q = reinterpret_cast<ushort *>(depth);
  for (int i = 0; i < width * height; i++)
    p[i] = q[i] / depth_factor_;

  // backprojection
  depth_map_device_->copyFrom(*depth_map_);
  backproject<float, Poly3CameraModel>(*depth_map_device_, *vertex_map_device_, model);
  vertex_map_->copyFrom(*vertex_map_device_);

  // set the depth factor
  depth_factor_ = factor;

  // for each object
  for(int i = 0; i < num_roi; i++)
  {
    int objID = int(rois[i * channel_roi + 1]);
    if (objID <= 0)
      continue;

    // pose
    const float* pose = poses + i * 7;
    std::cout << "quaternion " << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << std::endl;
    Eigen::Quaternionf quaternion(pose[0], pose[1], pose[2], pose[3]);
    Sophus::SE3f::Point translation(pose[4], pose[5], pose[6]);
    Sophus::SE3f T_co(quaternion, translation);

    /*
    // set labels
    label_indexes_.clear();
    for (int j = 0; j < width * height; j++)
    {
      if (labelmap[j] == objID)
      {
        label_indexes_.push_back(j);

        int x = j % width;
        int y = j / width;
        if ((*depth_map_)(x, y) > 0)
        {
          Vec3 dpoint = (*vertex_map_)(x, y);
          geometry_msgs::Point32 point;
          point.x = dpoint(0);
          point.y = dpoint(1);
          point.z = dpoint(2);
          output_points[objID - 1].push_back(point);
        }
      }
    }

    if (label_indexes_.size() < 400)
    {
      std::cout << "class id: " << objID << ", pixels: " << label_indexes_.size() << std::endl;
      continue;
    }

    // compute object center using depth and vertmap
    float Tz = 0;
    int c = 0;
    std::vector<float> ds;
    for (int j = 0; j < label_indexes_.size(); j++)
    {
      int x = label_indexes_[j] % width;
      int y = label_indexes_[j] / width;

      if ((*depth_map_)(x, y) > 0)
      {
        ds.push_back((*depth_map_)(x, y));
        Tz += (*depth_map_)(x, y);
        c++;
      }
    }

    float rx = 0;
    float ry = 0;
    if (pose[6])
    {
      rx = pose[4] / pose[6];
      ry = pose[5] / pose[6];
    }
    if (c > 0)
    {
      if (0)
      {
        size_t n = ds.size() / 2;
        std::nth_element(ds.begin(), ds.begin()+n, ds.end());
        Tz = ds[n];
      }
      else
        Tz /= c;

      // modify translation
      T_co.translation()(0) = rx * Tz;
      T_co.translation()(1) = ry * Tz;
      T_co.translation()(2) = Tz;
      std::cout << "Translation " << T_co.translation()(0) << " " << T_co.translation()(1) << " " << T_co.translation()(2) << std::endl;
    }
    else
      Tz = T_co.translation()(2);
    */

    // copy results
    outputs[i * 7 + 0] = T_co.unit_quaternion().w();
    outputs[i * 7 + 1] = T_co.unit_quaternion().x();
    outputs[i * 7 + 2] = T_co.unit_quaternion().y();
    outputs[i * 7 + 3] = T_co.unit_quaternion().z();
    outputs[i * 7 + 4] = T_co.translation()(0);
    outputs[i * 7 + 5] = T_co.translation()(1);
    outputs[i * 7 + 6] = T_co.translation()(2);
  }

  visualizePose(height, width, fx, fy, px, py, znear, zfar, rois, outputs, num_roi, channel_roi);
}



void Synthesizer::solveICP(const int* labelmap, unsigned char* depth, int height, int width, float fx, float fy, float px, float py, 
  float znear, float zfar, float factor, int num_roi, int channel_roi, const float* rois, const float* poses, 
  float* outputs, float* outputs_icp, float maxError)
{
  int iterations;
  if (setup_ == 0)
    setup(width, height);

  // build the camera paramters
  Eigen::Matrix<float,7,1,Eigen::DontAlign> params;
  params[0] = fx;
  params[1] = fy;
  params[2] = px;
  params[3] = py;
  params[4] = 0;
  params[5] = 0;
  params[6] = 0;
  df::Poly3CameraModel<float> model(params);

  DataForOpt data;
  data.width = width;
  data.height = height;
  data.labelmap = labelmap;
  data.label_indexes = &label_indexes_;
  data.depthRange = Eigen::Vector2f(znear, zfar);
  data.vertex_map = vertex_map_;
  data.predicted_verts = predicted_verts_;
  data.model = &model;

  // set the depth factor
  depth_factor_ = factor;

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, -fy, px+0.5, height-(py+0.5), znear, zfar);
  renderer_->setProjectionMatrix(projectionMatrix);
  renderer_vn_->setProjectionMatrix(projectionMatrix);

  // for each object
  for(int i = 0; i < num_roi; i++)
  {
    int objID = int(rois[i * channel_roi + 1]);
    data.objID = objID;
    if (objID <= 0)
      continue;

    // pose
    const float* pose = poses + i * 7;
    std::cout << "quaternion " << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3] << std::endl;
    Eigen::Quaternionf quaternion(pose[0], pose[1], pose[2], pose[3]);
    Sophus::SE3f::Point translation(pose[4], pose[5], pose[6]);
    Sophus::SE3f T_co(quaternion, translation);

    // render vertmap
    std::vector<Eigen::Matrix4f> transforms(1);
    std::vector<std::vector<pangolin::GlBuffer *> > attributeBuffers_vertmap(1);
    std::vector<pangolin::GlBuffer*> modelIndexBuffers(1);

    transforms[0] = T_co.matrix().cast<float>();
    attributeBuffers_vertmap[0].push_back(&texturedVertices_[objID - 1]);
    attributeBuffers_vertmap[0].push_back(&canonicalVertices_[objID - 1]);
    modelIndexBuffers[0] = &texturedIndices_[objID - 1];

    glClearColor(std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""));
    renderer_->render(attributeBuffers_vertmap, modelIndexBuffers, transforms);

    std::vector<float3> vertmap(width * height);
    renderer_->texture(0).Download(vertmap.data(), GL_RGB, GL_FLOAT);

    // render 3D points and normals
    std::vector<pangolin::GlBuffer *> attributeBuffers({&texturedVertices_[objID - 1], &vertexNormals_[objID - 1]});
    renderer_vn_->setModelViewMatrix(T_co.matrix().cast<float>());
    renderer_vn_->render(attributeBuffers, texturedIndices_[objID - 1], GL_TRIANGLES);

    const pangolin::GlTextureCudaArray & vertTex = renderer_vn_->texture(0);
    const pangolin::GlTextureCudaArray & normTex = renderer_vn_->texture(1);

    // copy predicted normals
    {
      pangolin::CudaScopedMappedArray scopedArray(normTex);
      cudaMemcpy2DFromArray(predicted_normals_device_->data(), normTex.width*4*sizeof(float), *scopedArray, 0, 0, normTex.width*4*sizeof(float), normTex.height, cudaMemcpyDeviceToDevice);
      predicted_normals_->copyFrom(*predicted_normals_device_);
    }

    // copy predicted vertices
    {
      pangolin::CudaScopedMappedArray scopedArray(vertTex);
      cudaMemcpy2DFromArray(predicted_verts_device_->data(), vertTex.width*4*sizeof(float), *scopedArray, 0, 0, vertTex.width*4*sizeof(float), vertTex.height, cudaMemcpyDeviceToDevice);
      predicted_verts_->copyFrom(*predicted_verts_device_);
    }

    // convert depth values
    label_indexes_.clear();
    float* p = depth_map_->data();
    ushort* q = reinterpret_cast<ushort *>(depth);
    for (int j = 0; j < width * height; j++)
    {
      if (labelmap[j] == objID)
      {
        p[j] = q[j] / depth_factor_;
        label_indexes_.push_back(j);
      }
      else
        p[j] = 0;
    }

    if (label_indexes_.size() < 400)
    {
      std::cout << "class id: " << objID << ", pixels: " << label_indexes_.size() << std::endl;
      continue;
    }

    // backprojection
    depth_map_device_->copyFrom(*depth_map_);
    backproject<float, Poly3CameraModel>(*depth_map_device_, *vertex_map_device_, model);
    vertex_map_->copyFrom(*vertex_map_device_);

    // compute object center using depth and vertmap
    float Tx = 0;
    float Ty = 0;
    float Tz = 0;
    int c = 0;
    std::vector<PointT> depth_points;
    std::vector<Vec3> model_points;
    for (int j = 0; j < label_indexes_.size(); j++)
    {
      int x = label_indexes_[j] % width;
      int y = label_indexes_[j] / width;

      if ((*depth_map_)(x, y) > 0)
      {
        float vx = vertmap[y * width + x].x - std::round(vertmap[y * width + x].x);
        float vy = vertmap[y * width + x].y;
        float vz = vertmap[y * width + x].z;

        if (std::isnan(vx) == 0 && std::isnan(vy) == 0 && std::isnan(vz) == 0)
        {
          Eigen::UnalignedVec4<float> normal = (*predicted_normals_)(x, y);
          Eigen::UnalignedVec4<float> vertex = (*predicted_verts_)(x, y);
          Vec3 dpoint = (*vertex_map_)(x, y);
          float error = normal.head<3>().dot(dpoint - vertex.head<3>());
          if (fabs(error) < maxError)
          {
            Tx += (dpoint(0) - vx);
            Ty += (dpoint(1) - vy);
            Tz += (dpoint(2) - vz);
            c++;
          }

          Vec3 mt;
          mt(0) = vx;
          mt(1) = vy;
          mt(2) = vz;
          model_points.push_back(mt);

          PointT pt;
          pt.x = dpoint(0);
          pt.y = dpoint(1);
          pt.z = dpoint(2);
          depth_points.push_back(pt);
        }
      }
    }

    float rx = 0;
    float ry = 0;
    if (pose[6])
    {
      rx = pose[4] / pose[6];
      ry = pose[5] / pose[6];
    }
    if (c > 0)
    {
      Tx /= c;
      Ty /= c;
      Tz /= c;

      // modify translation
      T_co.translation()(0) = rx * Tz;
      T_co.translation()(1) = ry * Tz;
      T_co.translation()(2) = Tz;
      std::cout << "Translation " << T_co.translation()(0) << " " << T_co.translation()(1) << " " << T_co.translation()(2) << std::endl;

      iterations = 50;
      refinePose(width, height, objID, znear, zfar, labelmap, data, model, T_co, iterations, maxError, 0);
      Tx = T_co.translation()(0);
      Ty = T_co.translation()(1);
      Tz = T_co.translation()(2);
      rx = Tx / Tz;
      ry = Ty / Tz;

      std::cout << "Translation after " << Tx << " " << Ty << " " << Tz << std::endl;
    }
    else
      Tz = T_co.translation()(2);

    // copy results
    outputs[i * 7 + 0] = T_co.unit_quaternion().w();
    outputs[i * 7 + 1] = T_co.unit_quaternion().x();
    outputs[i * 7 + 2] = T_co.unit_quaternion().y();
    outputs[i * 7 + 3] = T_co.unit_quaternion().z();
    outputs[i * 7 + 4] = T_co.translation()(0);
    outputs[i * 7 + 5] = T_co.translation()(1);
    outputs[i * 7 + 6] = T_co.translation()(2);

    // pose hypotheses
    std::vector<Sophus::SE3f> hyps;

    hyps.push_back(T_co);

    T_co.translation()(2) = Tz - 0.02;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz - 0.01;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz + 0.01;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz + 0.02;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz + 0.03;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz + 0.04;
    hyps.push_back(T_co);

    T_co.translation()(2) = Tz + 0.05;
    hyps.push_back(T_co);
    
    iterations = 8;
    for (int j = 0; j < hyps.size(); j++)
    {
      refinePose(width, height, objID, znear, zfar, labelmap, data, model, hyps[j], iterations, maxError, 1);
      std::cout << "pose " << j << std::endl << hyps[j].matrix() << std::endl;
    }

    if (depth_points.size() > 0)
    {
      // build a kd-tree of the depth points
      PointCloud::Ptr cloud(new PointCloud);
      cloud->width = depth_points.size();
      cloud->height = 1;
      cloud->points.resize(cloud->width * cloud->height);
      for (size_t j = 0; j < cloud->points.size(); j++)
      {
        cloud->points[j].x = depth_points[j].x;
        cloud->points[j].y = depth_points[j].y;
        cloud->points[j].z = depth_points[j].z;
      }
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(cloud);

      // use the metric in SegICP
      float max_score = -FLT_MAX;
      int choose = -1;
      for (int j = 0; j < hyps.size(); j++)
      {
        float score = 0;
        std::vector<int> flags(depth_points.size(), 0);
        #pragma omp parallel for
        for (int k = 0; k < model_points.size(); k++)
        {
          Vec3 pt = hyps[j] * model_points[k];
          PointT searchPoint;
          searchPoint.x = pt(0);
          searchPoint.y = pt(1);
          searchPoint.z = pt(2);

          std::vector<int> pointIdxRadiusSearch;
          std::vector<float> pointRadiusSquaredDistance;
          float radius = 0.01;
          if (kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
          {
            if (flags[pointIdxRadiusSearch[0]] == 0)
            {
              flags[pointIdxRadiusSearch[0]] = 1;
              score++;
            }
          }
        }
        score /= model_points.size();
        if (score > max_score)
        {
          max_score = score;
          choose = j;
        }
        printf("hypothesis %d, score %f\n", j, score);
      }
      printf("select hypothesis %d\n", choose);
      T_co = hyps[choose];
    }
    else
      T_co = hyps[0];

    // set output
    Eigen::Quaternionf quaternion_new = T_co.unit_quaternion();
    Sophus::SE3f::Point translation_new = T_co.translation();

    outputs_icp[i * 7 + 0] = quaternion_new.w();
    outputs_icp[i * 7 + 1] = quaternion_new.x();
    outputs_icp[i * 7 + 2] = quaternion_new.y();
    outputs_icp[i * 7 + 3] = quaternion_new.z();
    outputs_icp[i * 7 + 4] = translation_new(0);
    outputs_icp[i * 7 + 5] = translation_new(1);
    outputs_icp[i * 7 + 6] = translation_new(2);
  }

  visualizePose(height, width, fx, fy, px, py, znear, zfar, rois, outputs, num_roi, channel_roi);
}


void Synthesizer::visualizePose(int height, int width, float fx, float fy, float px, float py, float znear, float zfar, const float* rois, float* outputs, int num_roi, int channel_roi)
{
  if (setup_ == 0)
    setup(width, height);

  pangolin::OpenGlMatrixSpec projectionMatrix = pangolin::ProjectionMatrixRDF_TopLeft(width, height, fx, fy, px+0.5, py+0.5, znear, zfar);

  // render color image
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  glColor3ub(255,255,255);
  gtView_->ActivateScissorAndClear();
  float threshold = 0.1;

  for (int i = 0; i < num_roi; i++)
  {
    if ((outputs[i * 7 + 0] * outputs[i * 7 + 0] + outputs[i * 7 + 1] * outputs[i * 7 + 1] +
        outputs[i * 7 + 2] * outputs[i * 7 + 2] + outputs[i * 7 + 3] * outputs[i * 7 + 3]) == 0)
      continue;

    Eigen::Quaternionf quaternion(outputs[i * 7 + 0], outputs[i * 7 + 1], outputs[i * 7 + 2], outputs[i * 7 + 3]);
    Sophus::SE3f::Point translation(outputs[i * 7 + 4], outputs[i * 7 + 5], outputs[i * 7 + 6]);
    const Sophus::SE3f T_co(quaternion, translation);

    int class_id = int(rois[i * channel_roi + 1]) - 1;

    glMatrixMode(GL_PROJECTION);
    projectionMatrix.Load();
    glMatrixMode(GL_MODELVIEW);

    Eigen::Matrix4f mv = T_co.cast<float>().matrix();
    pangolin::OpenGlMatrix mvMatrix(mv);
    mvMatrix.Load();

    if (is_textured_[class_id])
    {
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
    else
    {
      glEnableClientState(GL_VERTEX_ARRAY);
      texturedVertices_[class_id].Bind();
      glVertexPointer(3,GL_FLOAT,0,0);
      glEnableClientState(GL_COLOR_ARRAY);
      vertexColors_[class_id].Bind();
      glColorPointer(3,GL_FLOAT,0,0);
      texturedIndices_[class_id].Bind();
      glDrawElements(GL_TRIANGLES, texturedIndices_[class_id].num_elements, GL_UNSIGNED_INT, 0);
      texturedIndices_[class_id].Unbind();
      texturedVertices_[class_id].Unbind();
      vertexColors_[class_id].Unbind();
      glDisableClientState(GL_VERTEX_ARRAY);
      glDisableClientState(GL_COLOR_ARRAY);
    }
  }

  pangolin::FinishFrame();
  // std::string filename = std::to_string(counter_++);
  // pangolin::SaveWindowOnRender(filename);
}


static double optEnergy(const std::vector<double> &pose, std::vector<double> &grad, void *data)
{
  DataForOpt* dataForOpt = (DataForOpt*) data;

  // SE3
  Eigen::Quaternionf quaternion(pose[0], pose[1], pose[2], pose[3]);
  Sophus::SE3f::Point translation(pose[4], pose[5], pose[6]);
  const Sophus::SE3f T_co(quaternion, translation);

  // compute point-wise distance
  int c = 0;
  float distance = 0;
  int width = dataForOpt->width;
  int height = dataForOpt->height;
  int objID = dataForOpt->objID;
  const int* labelmap = dataForOpt->labelmap;
  df::ManagedHostTensor2<Eigen::UnalignedVec4<float> >* predicted_verts = dataForOpt->predicted_verts;
  df::ManagedHostTensor2<Vec3>* vertex_map = dataForOpt->vertex_map;
  Eigen::Vector2f depthRange = dataForOpt->depthRange;

  for (int i = 0; i < dataForOpt->label_indexes->size(); i++)
  {
    int x = (*dataForOpt->label_indexes)[i] % width;
    int y = (*dataForOpt->label_indexes)[i] / width;

    float px = (*predicted_verts)(x, y)(0);
    float py = (*predicted_verts)(x, y)(1);
    float pz = (*predicted_verts)(x, y)(2);

    Sophus::SE3f::Point point(px, py, pz);
    Sophus::SE3f::Point point_new = T_co * point;

    px = point_new(0);
    py = point_new(1);
    pz = point_new(2);

    float vx = (*vertex_map)(x, y)(0);
    float vy = (*vertex_map)(x, y)(1);
    float vz = (*vertex_map)(x, y)(2);
    if (std::isnan(px) == 0 && std::isnan(py) == 0 && std::isnan(pz) == 0 && vz > depthRange(0) && vz < depthRange(1) && pz > depthRange(0) && pz < depthRange(1))
    {
      distance += std::sqrt((px - vx) * (px - vx) + (py - vy) * (py - vy) + (pz - vz) * (pz - vz));
      c++;
    }
  }
  if (c)
    distance /= c;
  float energy = distance;

  return energy;
}


double Synthesizer::poseWithOpt(std::vector<double> & vec, DataForOpt data, int iterations)
{
  // set up optimization algorithm (gradient free)
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 7); 

  // set optimization bounds 
  double rotRange = 0.1;
  double tRangeXY = 0.01;
  double tRangeZ = 0.1; // pose uncertainty is larger in Z direction
	
  std::vector<double> lb(7);
  lb[0] = vec[0] - rotRange;
  lb[1] = vec[1] - rotRange;
  lb[2] = vec[2] - rotRange;
  lb[3] = vec[3] - rotRange;
  lb[4] = vec[4] - tRangeXY;
  lb[5] = vec[5] - tRangeXY;
  lb[6] = vec[6] - tRangeZ;
  opt.set_lower_bounds(lb);
      
  std::vector<double> ub(7);
  ub[0] = vec[0] + rotRange;
  ub[1] = vec[1] + rotRange;
  ub[2] = vec[2] + rotRange;
  ub[3] = vec[3] + rotRange;
  ub[4] = vec[4] + tRangeXY;
  ub[5] = vec[5] + tRangeXY;
  ub[6] = vec[6] + tRangeZ;
  opt.set_upper_bounds(ub);
      
  // configure NLopt
  opt.set_min_objective(optEnergy, &data);
  opt.set_maxeval(iterations);

  // run optimization
  double energy;
  nlopt::result result = opt.optimize(vec, energy);

  // std::cout << "distance after optimization: " << energy << std::endl;
   
  return energy;
}
