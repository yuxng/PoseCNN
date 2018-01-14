#include "kinect_fusion.hpp"

using namespace df;

KinectFusion::KinectFusion(std::string rig_specification_file)
{
  setup_cameras(rig_specification_file);
  create_window();
  std::cout << "created window" << std::endl;
  create_tensors();
  std::cout << "created tensors" << std::endl;
}

// setup cameras
void KinectFusion::setup_cameras(std::string rig_specification_file)
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

  T_dc_ = rig_->transformCameraToRig(depth_stream_index).inverse() * rig_->transformCameraToRig(color_stream_index);
  std::cout << "T_dc: " << T_dc_.matrix() << std::endl;

  colorKinv_ << 1. / color_camera_->params()[0], 0, -color_camera_->params()[2] / color_camera_->params()[0],
                0, 1. / color_camera_->params()[1], -color_camera_->params()[3] / color_camera_->params()[1],
                0, 0, 1;
  depthKinv_ << 1. / depth_camera_->params()[0], 0, -depth_camera_->params()[2] / depth_camera_->params()[0],
                0, 1. / depth_camera_->params()[1], -depth_camera_->params()[3] / depth_camera_->params()[1],
                0, 0, 1;
}

// create window
void KinectFusion::create_window()
{
  pangolin::CreateWindowAndBind("Kinect Fusion", 1280, 960);

  depthCamState_ = new pangolin::OpenGlRenderState(ProjectionMatrixRDF_TopLeft(*depth_camera_, 0.01, 1000.0));
  pangolin::Handler3D camHandler(*depthCamState_);
  disp3d_ = &pangolin::Display("disp3d").SetAspect(640.0/480.).SetHandler(&camHandler);

  colorCamState_ = new pangolin::OpenGlRenderState(ProjectionMatrixRDF_TopLeft(*color_camera_, 0.01, 1000.0));
  colorView_ = &pangolin::Display("color");
  depthView_ = &pangolin::Display("depth");
  labelView_ = &pangolin::Display("label");

  pangolin::CreatePanel("panel").SetBounds(0, 1, 0, pangolin::Attach::Pix(0));
  allView_ = &pangolin::Display("multi").AddDisplay(*disp3d_).AddDisplay(*colorView_).AddDisplay(*depthView_).AddDisplay(*labelView_)
            .SetBounds(0, 1, pangolin::Attach::Pix(0), 1).SetLayout(pangolin::LayoutEqual);

  // texture
  colorTex_ = new pangolin::GlTexture(color_camera_->width(), color_camera_->height());
  depthTex_ = new pangolin::GlTexture(depth_camera_->width(), depth_camera_->height());
  labelTex_ = new pangolin::GlTexture(depth_camera_->width(), depth_camera_->height());

  colorView_->SetAspect(colorTex_->width/static_cast<double>(colorTex_->height));
  depthView_->SetAspect(depthTex_->width/static_cast<double>(depthTex_->height));
  labelView_->SetAspect(labelTex_->width/static_cast<double>(labelTex_->height));

  // lighting
  GLfloat lightPosition[] = { 0.0, 0.0, -1.0, 0.0 };
  GLfloat lightAmbient[] = { 0.0, 0.0, 0.0, 1.0 };
  glShadeModel(GL_SMOOTH);
  glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
  glLightfv(GL_LIGHT0,GL_AMBIENT, lightAmbient);
  glEnable(GL_LIGHT0);
  srand(pangolin::TimeNow_s());
  GLfloat modelDiffuseColor[] = { 0.95f, 0.95f, 1.0f, 1.f };
  GLfloat modelSpecularColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, modelDiffuseColor);
  glMaterialfv(GL_FRONT, GL_SPECULAR, modelSpecularColor);
  GLfloat shininess[] = {50};
  glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST);
}


// allocate tensors
void KinectFusion::create_tensors()
{
  // depth map
  depth_map_ = new ManagedTensor<2, float>({depth_camera_->width(), depth_camera_->height()});
  depth_map_device_ = new ManagedTensor<2, float, DeviceResident>(depth_map_->dimensions());
  depth_factor_ = 1000.0;
  depth_cutoff_ = 20.0;

  // probability
  probability_map_device_ = new ManagedDeviceTensor2<Vec>({depth_camera_->width(), depth_camera_->height()});

  // class colors
  class_colors_device_ = new ManagedDeviceTensor1<Vec3uc>(10);

  // color
  color_map_ = new ManagedHostTensor2<Vec3>({depth_camera_->width(), depth_camera_->height()});
  color_map_device_ = new ManagedDeviceTensor2<Vec3>(color_map_->dimensions());

  // labels
  labels_ = new ManagedHostTensor2<int>({depth_camera_->width(), depth_camera_->height()});
  labels_device_ = new ManagedDeviceTensor2<int>(labels_->dimensions());
  label_colors_ = new ManagedHostTensor2<Vec3uc>(labels_->dimensions());
  label_colors_device_ = new ManagedDeviceTensor2<Vec3uc>(labels_->dimensions());

  // 3D points
  vertex_map_ = new ManagedHostTensor2<Vec3>({depth_camera_->width(), depth_camera_->height()});
  vertex_map_device_ = new ManagedDeviceTensor2<Vec3>(vertex_map_->dimensions());

  // predicted vertices and normals
  predicted_verts_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({depth_camera_->width(), depth_camera_->height()});
  predicted_normals_ = new ManagedHostTensor2<Eigen::UnalignedVec4<float> >({depth_camera_->width(), depth_camera_->height()});

  predicted_verts_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > (predicted_verts_->dimensions());
  predicted_normals_device_ = new ManagedDeviceTensor2<Eigen::UnalignedVec4<float> > (predicted_normals_->dimensions());

  // in extract surface
  dVertices_ = new ManagedTensor<2, float, DeviceResident>({3,1});
  dWeldedVertices_ = new ManagedTensor<2, float, DeviceResident>({3,1});
  dIndices_ = new ManagedTensor<1, int, DeviceResident>(Eigen::Matrix<uint,1,1>(1));
  dNormals_ = new ManagedDeviceTensor1<Eigen::UnalignedVec3<float> >(1);
  // dColors_ = new ManagedTensor<2, unsigned char, DeviceResident>({3,1});
  dColors_ = new ManagedDeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> >(1);
  numUniqueVertices_ = 0;

  // for rendering
  vertBuffer_ = new pangolin::GlBufferCudaPtr(pangolin::GlArrayBuffer, dVertices_->dimensionSize(1), GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);
  normBuffer_ = new pangolin::GlBufferCudaPtr(pangolin::GlArrayBuffer, dNormals_->length(), GL_FLOAT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);
  indexBuffer_ = new pangolin::GlBufferCudaPtr(pangolin::GlElementArrayBuffer, dIndices_->dimensionSize(0), GL_INT, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);
  colorBuffer_ = new pangolin::GlBufferCudaPtr(pangolin::GlArrayBuffer, dVertices_->dimensionSize(1), GL_UNSIGNED_BYTE, 3, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);

  // voxels
  voxel_data_ = new ManagedTensor<3, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident>({512, 512, 512});
  float voxelGridOffsetX = -1;
  float voxelGridOffsetY = -1;
  float voxelGridOffsetZ = 0;
  float voxelGridDimX = 2;
  float voxelGridDimY = 2;
  float voxelGridDimZ = 2;
  voxel_grid_ = new DeviceVoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >(voxel_data_->dimensions(), voxel_data_->data(),
                                                      Eigen::AlignedBox3f(Eigen::Vector3f(voxelGridOffsetX, voxelGridOffsetY, voxelGridOffsetZ),
                                                                          Eigen::Vector3f(voxelGridOffsetX + voxelGridDimX, voxelGridOffsetY + voxelGridDimY, voxelGridOffsetZ + voxelGridDimZ)));

  voxel_grid_->fill(CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>::zero());
  initMarchingCubesTables();

  // renderer
  renderer_ = new GLRenderer<VertAndNormalRenderType>(depth_camera_->width(), depth_camera_->height());
  {
    // the +0.5 on principal point accounts for OpenGL assuming integral-valued image coordinates
    // land on the corners of pixels, which is inconsistent with the model used everywhere else,
    // which assumes integral-valued image coordinates land on the centers of pixels.
    // this has been tested to work by rendering a prediction and then reprojecting it using
    // the same camera parameters and ensuring the point at each location in the vertex map
    // projects on the corresponding pixel.
    pangolin::OpenGlRenderState rendererCam(pangolin::ProjectionMatrixRDF_TopLeft(depth_camera_->width(), depth_camera_->height(),
                                                                                  depth_camera_->params()[0], -depth_camera_->params()[1],
                                                                                  depth_camera_->params()[2]+0.5,
                                                                                  depth_camera_->height()-(depth_camera_->params()[3]+0.5), 0.25, 6.0));
    renderer_->setProjectionMatrix(rendererCam.GetProjectionMatrix());
    renderer_->setModelViewMatrix(voxel_grid_->gridToWorldTransform());
  }

  // ICP
  transformer_ = new RigidTransformer<float>;
}


// set the voxel grid size
void KinectFusion::set_voxel_grid(float voxelGridOffsetX, float voxelGridOffsetY, float voxelGridOffsetZ, float voxelGridDimX, float voxelGridDimY, float voxelGridDimZ)
{
  delete voxel_grid_;
  voxel_grid_ = new DeviceVoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >(voxel_data_->dimensions(), voxel_data_->data(),
                                                      Eigen::AlignedBox3f(Eigen::Vector3f(voxelGridOffsetX, voxelGridOffsetY, voxelGridOffsetZ),
                                                                          Eigen::Vector3f(voxelGridOffsetX + voxelGridDimX, voxelGridOffsetY + voxelGridDimY, voxelGridOffsetZ + voxelGridDimZ)));

  reset();
}

// estimate camera pose with ICP
void KinectFusion::solve_pose(float* pose_worldToLive, float* pose_liveToWorld)
{
  const Camera<Poly3CameraModel,double> * poly3DepthCamera = dynamic_cast<const Camera<Poly3CameraModel,double> *>(depth_camera_);
  if (!poly3DepthCamera)
    throw std::runtime_error("expected Poly3 model for the depth camera");
  Poly3CameraModel<float> model = poly3DepthCamera->model().cast<float>();

  Eigen::Vector2f depthRange(0.25, 6.0);
  float maxError = 0.02;
  uint icpIterations = 8;

  std::cout << transformer_->worldToLiveTransformation().matrix3x4() << std::endl << std::endl;

  Sophus::SE3f update = icp(*vertex_map_device_, *predicted_verts_device_, *predicted_normals_device_,
                                 model, transformer_->worldToLiveTransformation(),
                                 depthRange, maxError, icpIterations);

  transformer_->setWorldToLiveTransformation(update * transformer_->worldToLiveTransformation());
  std::cout << transformer_->worldToLiveTransformation().matrix3x4() << std::endl << std::endl;
  std::cout << transformer_->liveToWorldTransformation().matrix3x4() << std::endl << std::endl;

  // copy the pose
  if (pose_worldToLive && pose_liveToWorld)
  {
    float* m = &(transformer_->worldToLiveTransformation().matrix3x4()(0));
    int count = 0;
    for (int j = 0; j < 4; j++)
    {
      for (int i = 0; i < 3; i++)
        pose_worldToLive[i*4+j] = m[count++];
    }

    float* n = &(transformer_->liveToWorldTransformation().matrix3x4()(0));
    count = 0;
    for (int j = 0; j < 4; j++)
    {
      for (int i = 0; i < 3; i++)
        pose_liveToWorld[i*4+j] = n[count++];
    }
  }
}

// fuse depth
void KinectFusion::fuse_depth()
{
  const Camera<Poly3CameraModel,double> * poly3DepthCamera = dynamic_cast<const Camera<Poly3CameraModel,double> *>(depth_camera_);
  if (!poly3DepthCamera) 
    throw std::runtime_error("expected Poly3 model for the depth camera");
  Poly3CameraModel<float> depthModel = poly3DepthCamera->model().cast<float>();

  const Camera<Poly3CameraModel,double> * poly3ColorCamera = dynamic_cast<const Camera<Poly3CameraModel,double> *>(color_camera_);
  if (!poly3ColorCamera)
    throw std::runtime_error("expected Poly3 model for the color camera");
  Poly3CameraModel<float> colorModel = poly3ColorCamera->model().cast<float>();

  float truncationDistance = 0.04;
  df::fuseFrame(*voxel_grid_, *transformer_, depthModel, colorModel, T_dc_.inverse().cast<float>(), *depth_map_device_, truncationDistance, 
                internal::FusionTypeTraits<ProbabilityVoxel>::PackedInput<float>(truncationDistance, 100.f, *probability_map_device_));
}

// extract surface
void KinectFusion::extract_surface(int* labels_return)
{
  extractSurface(*dVertices_, *voxel_grid_, 0.02f);

  dWeldedVertices_->resize(dVertices_->dimensions());
  dIndices_->resize(Eigen::Matrix<uint,1,1>(dVertices_->dimensionSize(1)));
  numUniqueVertices_ = weldVertices(*dVertices_, *dWeldedVertices_, *dIndices_);

  std::cout << numUniqueVertices_ << " unique vertices / " << dVertices_->dimensionSize(1) << std::endl;

  dNormals_->resize(numUniqueVertices_);
  DeviceTensor2<float> dNormals__( {3, dNormals_->length() }, reinterpret_cast<float *>(dNormals_->data()));
  Tensor<2, float, DeviceResident> actualWeldedVertices(dNormals__.dimensions(), dWeldedVertices_->data());

  computeSignedDistanceGradientNormals(actualWeldedVertices, dNormals__, *voxel_grid_);

  // compute labels
  const Camera<Poly3CameraModel,double> * poly3DepthCamera = dynamic_cast<const Camera<Poly3CameraModel,double> *>(depth_camera_);
  if (!poly3DepthCamera) 
    throw std::runtime_error("expected Poly3 model for the depth camera");
  Poly3CameraModel<float> depthModel = poly3DepthCamera->model().cast<float>();
  
  computeLabels(*transformer_, depthModel, *depth_map_device_, *voxel_grid_, *labels_device_, *label_colors_device_, *class_colors_device_);
  labels_->copyFrom(*labels_device_);
  if(labels_return)
    memcpy(labels_return, labels_->data(), sizeof(int) * labels_->dimensionSize(0) * labels_->dimensionSize(1));
  label_colors_->copyFrom(*label_colors_device_);
  label_texture()->Upload(reinterpret_cast<unsigned char *>(label_colors_->data()), GL_RGB, GL_UNSIGNED_BYTE);

  // compute vertex color
  DeviceTensor1<Vec3> actualWeldedVertices_( dNormals_->length(), reinterpret_cast<Vec3 *>(actualWeldedVertices.data()) );
  dColors_->resize(actualWeldedVertices_.length());
  computeSurfaceColors(actualWeldedVertices_, *dColors_, *voxel_grid_, *class_colors_device_);

  // copy colors
  colorBuffer_->Resize(numUniqueVertices_);
  {
    pangolin::CudaScopedMappedPtr scopedPtr(*colorBuffer_);
    checkCuda(cudaMemcpy(*scopedPtr, dColors_->data(), dColors_->length()*sizeof(Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign>), cudaMemcpyDeviceToDevice));
  }

  cudaDeviceSynchronize();
  CheckCudaDieOnError();
}


// rendering
void KinectFusion::render()
{
  // copy indexes
  indexBuffer_->Resize(dIndices_->dimensionSize(0));
  {
    pangolin::CudaScopedMappedPtr scopedPtr(*indexBuffer_);
    checkCuda( cudaMemcpy(*scopedPtr, dIndices_->data(), dIndices_->count()*sizeof(int), cudaMemcpyDeviceToDevice) );
  }

  // copy vertices
  vertBuffer_->Resize(numUniqueVertices_);
  {
    pangolin::CudaScopedMappedPtr scopedPtr(*vertBuffer_);
    checkCuda(cudaMemcpy(*scopedPtr, dWeldedVertices_->data(), numUniqueVertices_*3*sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // copy normals
  normBuffer_->Resize(dNormals_->length());
  {
    pangolin::CudaScopedMappedPtr scopedPtr(*normBuffer_);
    checkCuda(cudaMemcpy(*scopedPtr, dNormals_->data(), dNormals_->length()*3*sizeof(float), cudaMemcpyDeviceToDevice));
  }

  std::vector<pangolin::GlBuffer *> attributeBuffers({vertBuffer_, normBuffer_});

  renderer_->setModelViewMatrix(transformer_->worldToLiveTransformation().matrix() * voxel_grid_->gridToWorldTransform());
  renderer_->render(attributeBuffers, *indexBuffer_, GL_TRIANGLES);

  const pangolin::GlTextureCudaArray & vertTex = renderer_->texture(0);
  const pangolin::GlTextureCudaArray & normTex = renderer_->texture(1);

  cudaDeviceSynchronize();
  CheckCudaDieOnError();

  // copy predicted vertices
  {
    std::cout << vertTex.width << " x " << vertTex.height << std::endl;
    std::cout << vertTex.internal_format << std::endl;

    pangolin::CudaScopedMappedArray scopedArray(vertTex);
    cudaMemcpy2DFromArray(predicted_verts_device_->data(), vertTex.width*4*sizeof(float), *scopedArray, 0, 0, vertTex.width*4*sizeof(float), vertTex.height, cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
  CheckCudaDieOnError();

  // copy predicted normals
  {
    pangolin::CudaScopedMappedArray scopedArray(normTex);
    cudaMemcpy2DFromArray(predicted_normals_device_->data(), normTex.width*4*sizeof(float), *scopedArray, 0, 0, normTex.width*4*sizeof(float), normTex.height, cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
  CheckCudaDieOnError();
}


// draw model
void KinectFusion::draw(std::string filename, int flag)
{ 
  disp3d_->ActivateScissorAndClear(*depthCamState_);
  glColor3ub(255, 255, 255);
  colorTex_->RenderToViewportFlipY();
/*
  glPushMatrix();
  glMultMatrixX(transformer_->liveToWorldTransformation().matrix());

  glColor3ub(255, 0, 0);
  pangolin::glDrawFrustrum(colorKinv_, color_camera_->width(), color_camera_->height(), 0.1f);

  glColor3ub(0, 255, 0);
  Eigen::Matrix4f T = T_dc_.matrix().cast<float>();
  pangolin::glDrawFrustrum(depthKinv_, depth_camera_->width(), depth_camera_->height(), T, 0.1f);

  // show point cloud
  vertex_map_->copyFrom(*vertex_map_device_);
  glColor3ub(128, 128, 128);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, vertex_map_->data());
  glDrawArrays(GL_POINTS, 0, depth_camera_->width() * depth_camera_->height());
  glDisableClientState(GL_VERTEX_ARRAY);
  glPopMatrix();

  // show bounding box
  glColor3ub(255, 0, 0);
  Eigen::AlignedBox3f bb = voxel_grid_->boundingBox();
  pangolin::glDrawAlignedBox(bb);

  // show predicted points
  predicted_verts_->copyFrom(*predicted_verts_device_);
  predicted_normals_->copyFrom(*predicted_normals_device_);
  glPushMatrix();
  glMultMatrixX(transformer_->liveToWorldTransformation().matrix());
  glColor3ub(0, 255, 0);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 4*sizeof(float), predicted_verts_->data());
  glEnableClientState(GL_NORMAL_ARRAY);
  glNormalPointer(GL_FLOAT, 4*sizeof(float), predicted_normals_->data());
  glDrawArrays(GL_POINTS, 0, predicted_verts_->count());
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glPopMatrix();
*/  

  // show color image
  colorView_->ActivateScissorAndClear();
  labelTex_->RenderToViewportFlipY();
  glColor3ub(255,255,255);
  colorTex_->RenderToViewportFlipY();

  Eigen::Matrix<float, 3, 3> R;
  R << 1, 0, 0,
       0, 1, 0,
       0, 0, 1;
  Eigen::Matrix<float, 3, 1> T;
  T << 0, 0, 4;

  Sophus::SE3<float> RT(R, T);
  RigidTransformer<float> transformer;
  transformer.setWorldToLiveTransformation(RT * transformer_->worldToLiveTransformation());

  // show overlay
  colorCamState_->SetModelViewMatrix(transformer.worldToLiveTransformation().matrix());
  colorCamState_->Apply();

  glClear(GL_DEPTH_BUFFER_BIT);
  glPushMatrix();
  glMultMatrixX(T_dc_.inverse().matrix());
  glVoxelGridCoords(*voxel_grid_);
  glEnable(GL_LIGHTING);
  glEnable(GL_NORMALIZE);
  renderModel(*vertBuffer_, *normBuffer_, *indexBuffer_, *colorBuffer_);
  glDisable(GL_LIGHTING);
  glPopMatrix();

  // show depth image
  depthView_->ActivateScissorAndClear();
  depthTex_->RenderToViewportFlipY();

  // show label image
  labelView_->ActivateScissorAndClear();
  labelTex_->RenderToViewportFlipY();
  colorCamState_->SetModelViewMatrix(transformer.worldToLiveTransformation().matrix());
  colorCamState_->Apply();
  glClear(GL_DEPTH_BUFFER_BIT);
  glPushMatrix();
  glMultMatrixX(T_dc_.inverse().matrix());
  glVoxelGridCoords(*voxel_grid_);
  glEnable(GL_LIGHTING);
  glEnable(GL_NORMALIZE);
  renderModel(*vertBuffer_, *normBuffer_, *indexBuffer_);
  glDisable(GL_LIGHTING);
  glPopMatrix();

  pangolin::FinishFrame();

  // save frame
  if (flag)
  {
    labelView_->SaveOnRender(filename + "_live");
    colorView_->SaveOnRender(filename + "_world");
  }
}


// backproject
void KinectFusion::back_project()
{
  const Camera<Poly3CameraModel,double> * poly3DepthCamera = dynamic_cast<const Camera<Poly3CameraModel,double> *>(depth_camera_);
  if (!poly3DepthCamera)
    throw std::runtime_error("expected Poly3 model for the depth camera");

  Poly3CameraModel<float> model = poly3DepthCamera->model().cast<float>();
  depth_map_device_->copyFrom(*depth_map_);
  backproject<float, Poly3CameraModel>(*depth_map_device_, *vertex_map_device_, model);

  std::cout << model.focalLength().transpose() << std::endl;
  std::cout << model.principalPoint().transpose() << std::endl << std::endl;
}


// render model
void KinectFusion::renderModel(pangolin::GlBufferCudaPtr & vertBuffer, pangolin::GlBufferCudaPtr & normBuffer, pangolin::GlBufferCudaPtr & indexBuffer, pangolin::GlBufferCudaPtr & colorBuffer)
{
  vertBuffer.Bind();
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3,GL_FLOAT,0,0);

  colorBuffer.Bind();
  glEnableClientState(GL_COLOR_ARRAY);
  glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);

  normBuffer.Bind();
  glEnableClientState(GL_NORMAL_ARRAY);
  glNormalPointer(GL_FLOAT,0,0);

  indexBuffer.Bind();
  glDrawElements(GL_TRIANGLES,indexBuffer.num_elements,GL_UNSIGNED_INT,0);

  indexBuffer.Unbind();;
  vertBuffer.Unbind();
  normBuffer.Unbind();
  colorBuffer.Unbind();
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}


// render model
void KinectFusion::renderModel(pangolin::GlBufferCudaPtr & vertBuffer, pangolin::GlBufferCudaPtr & normBuffer, pangolin::GlBufferCudaPtr & indexBuffer)
{
  vertBuffer.Bind();
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3,GL_FLOAT,0,0);

  normBuffer.Bind();
  glEnableClientState(GL_NORMAL_ARRAY);
  glNormalPointer(GL_FLOAT,0,0);

  indexBuffer.Bind();
  glDrawElements(GL_TRIANGLES,indexBuffer.num_elements,GL_UNSIGNED_INT,0);

  indexBuffer.Unbind();;
  vertBuffer.Unbind();
  normBuffer.Unbind();
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);
}


// feed data
void KinectFusion::feed_data(unsigned char* depth, unsigned char* color, int width, int height, float factor)
{
  // set the depth factor
  depth_factor_ = factor;

  // convert depth values
  float* p = depth_map()->data();

  ushort* q = reinterpret_cast<ushort *>(depth);
  for (int i = 0; i < width * height; i++)
    p[i] = q[i] / depth_factor_;

  color_texture()->Upload(color, GL_RGB, GL_UNSIGNED_BYTE);
  depth_texture()->Upload(depth_map()->data(), GL_LUMINANCE, GL_FLOAT);
}


// feed predicted labels
void KinectFusion::feed_label(unsigned char* im_label, float* probability, unsigned char* colors)
{
  // label_texture()->Upload(im_label, GL_RGB, GL_UNSIGNED_BYTE);
  depth_texture()->Upload(im_label, GL_RGB, GL_UNSIGNED_BYTE);

  // process color
  ConstHostTensor2<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > colorImage({depth_camera_->width(), depth_camera_->height()}, 
    reinterpret_cast<const Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> *>(im_label));

  std::transform(colorImage.data(), colorImage.data() + colorImage.count(),
                 color_map_->data(), [](const Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> & c) {
                 return c.cast<float>()*(1/float(255));
  });

  color_map_device_->copyFrom(*color_map_);

  // process probability
  HostTensor2<Eigen::Matrix<float,10,1,Eigen::DontAlign> > probImage({depth_camera_->width(), depth_camera_->height()}, 
    reinterpret_cast<Eigen::Matrix<float,10,1,Eigen::DontAlign> *>(probability));

  probability_map_device_->copyFrom(probImage);

  // class colors
  HostTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > class_color({10}, 
    reinterpret_cast<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> *>(colors));

  class_colors_device_->copyFrom(class_color);
}


// reset voxels
void KinectFusion::reset()
{
  voxel_grid_->fill(CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>::zero());
  delete transformer_;
  transformer_ = new RigidTransformer<float>;
}


// save reconstructed model
void KinectFusion::save_model(std::string filename)
{
  ManagedHostTensor1<Vec3> hWeldedVertices(numUniqueVertices_);
  hWeldedVertices.copyFrom(DeviceTensor1<Vec3>(numUniqueVertices_, reinterpret_cast<Vec3*>(dWeldedVertices_->data())));

  const int nFaces = dIndices_->length() / 3;
  ManagedHostTensor1<Vec3i> hIndices(nFaces);
  hIndices.copyFrom(DeviceTensor1<Vec3i>(nFaces, reinterpret_cast<Vec3i *>(dIndices_->data())));

  std::ofstream meshStream(filename);
  meshStream << "ply" << std::endl;
  meshStream << "format ascii 1.0" << std::endl;

  meshStream << "element vertex " << numUniqueVertices_ << std::endl;
  meshStream << "property float32 x" << std::endl;
  meshStream << "property float32 y" << std::endl;
  meshStream << "property float32 z" << std::endl;
  meshStream << "element face " << nFaces << std::endl;
  meshStream << "property list uint8 int32 vertex_index" << std::endl;
  meshStream << "end_header" << std::endl;

  for (int i = 0; i < numUniqueVertices_; i++)
  {
    const Vec3 & vGrid = hWeldedVertices(i);
    const Vec3 vWorld = voxel_grid_->gridToWorld(vGrid);
    meshStream << vWorld(0) << " " << vWorld(1) << " " << vWorld(2) << std::endl;
  }

  for (int i = 0; i < nFaces; ++i)
  {
    const Vec3i & face = hIndices(i);
    meshStream << "3 " << face(2) << " " << face(1) << " " << face(0) << std::endl;
  }
}

int main(int argc, char * * argv) 
{
  std::string rigSpecificationFile, inputString;
  int colorStreamIndex = 0;
  int depthStreamIndex = 1;

  // parse arguments
  {
    bool switchStreams = false;

    OptParse optParse;
    optParse.registerOption("rig",rigSpecificationFile,'r',true);
    optParse.registerOption("input",inputString,'i',true);
    optParse.registerOption("switchStreams",switchStreams);
    optParse.parseOptions(argc,argv);

    if (switchStreams) 
    {
      colorStreamIndex = depthStreamIndex;
      depthStreamIndex = 1-colorStreamIndex;
    }
  }

  pangolin::VideoInput video(inputString);
  pangolin::VideoPlaybackInterface * playback = video.Cast<pangolin::VideoPlaybackInterface>();

  // check video
  if (video.Streams().size() != 2) 
    throw std::runtime_error("expecting two-stream video (RGB + depth)");

  const pangolin::StreamInfo & colorStreamInfo = video.Streams()[colorStreamIndex];
  const pangolin::StreamInfo & depthStreamInfo = video.Streams()[depthStreamIndex];

  std::cout << "color: " << colorStreamInfo.Width() << " x " << colorStreamInfo.Height() << std::endl;
  std::cout << "depth: " << depthStreamInfo.Width() << " x " << depthStreamInfo.Height() << std::endl;

  std::vector<unsigned char> videoBuffer(colorStreamInfo.SizeBytes() + depthStreamInfo.SizeBytes());

  pangolin::GlPixFormat colorPixFormat(colorStreamInfo.PixFormat());
  pangolin::GlPixFormat depthPixFormat(depthStreamInfo.PixFormat());

  std::cout << "color format: " << colorStreamInfo.PixFormat().operator std::string() << std::endl;
  std::cout << "depth format: " << depthStreamInfo.PixFormat().operator std::string() << std::endl;

  // create kinect fusion
  KinectFusion KF(rigSpecificationFile);

  bool frameCurrent = false;
  bool vertMapCurrent = false;
  bool havePrediction = false;

  // main loop
  while (!pangolin::ShouldQuit()) 
  {
    if (pangolin::HasResized())
      pangolin::DisplayBase().ActivateScissorAndClear();

    // process a new frame
    if (!frameCurrent) 
    {
      GlobalTimer::tick("preprocess");

      bool success = video.GrabNext(videoBuffer.data());

      KF.feed_data(videoBuffer.data() + (uint64_t)depthStreamInfo.Offset(), videoBuffer.data() + (uint64_t)colorStreamInfo.Offset(), depthStreamInfo.Width(), depthStreamInfo.Height(), 10000.0);

      GlobalTimer::tock("preprocess");

      frameCurrent = true;
      vertMapCurrent = false;
    }

    // compute 3D points
    if (!vertMapCurrent) 
    {
      GlobalTimer::tick("backproject");
      std::cout << "backproject" << std::endl;
      KF.back_project();
      GlobalTimer::tock("backproject");
      vertMapCurrent = true;
    }

    // camera pose estimation
    if (havePrediction) 
    {
      GlobalTimer::tick("solve pose");
      std::cout << "solve pose" << std::endl;
      KF.solve_pose(NULL, NULL);
      GlobalTimer::tock("solve pose");
    }

    // fusion depth frame
    GlobalTimer::tick("fusion");
    std::cout << "fusion" << std::endl;
    KF.fuse_depth();
    GlobalTimer::tock("fusion");

    // extract surface
    GlobalTimer::tick("marching cubes");
    std::cout << "extract surface" << std::endl;
    KF.extract_surface(NULL);
    GlobalTimer::tock("marching cubes");

    // rendering
    GlobalTimer::tick("render");
    std::cout << "rendering" << std::endl;
    KF.render();
    havePrediction = true;
    GlobalTimer::tock("render");

    // drawing
    GlobalTimer::tick("rendering");
    KF.draw("test", 0);

    // move to next frame
    frameCurrent = false;

    GlobalTimer::tock("rendering");
  }
}
