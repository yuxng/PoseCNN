#include "rendering.hpp"

Render::Render(std::string model_file)
{
  counter_ = 0;
  loadModels(model_file);
}


Render::~Render()
{
  for(int i = 0; i < models_.size(); i++)
  {
    free(models_[i]->vertices);
    free(models_[i]->faces);
    free(models_[i]);
  }
}

void Render::setup(std::string model_file)
{
  counter_ = 0;
  loadModels(model_file);
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
  models_.resize(num_models);
  texture_names_.resize(num_models);

  for (int m = 0; m < num_models; ++m)
  {
    models_[m] = new MyModel();
    loadTexturedMesh(model_names[m], texture_names_[m], models_[m]);
    std::cout << texture_names_[m] << std::endl;
  }
}

void Render::loadTexturedMesh(const std::string filename, std::string & texture_name, MyModel* model)
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

    // constrcut the model
    model->num_vertices = assimpMesh->mNumVertices;
    model->num_faces = assimpMesh->mNumFaces;
    model->vertices = malloc( assimpMesh->mNumVertices*sizeof(float)*3 );
    memcpy(model->vertices, assimpMesh->mVertices, assimpMesh->mNumVertices*sizeof(float)*3);

    std::vector<uint3> faces3(assimpMesh->mNumFaces);
    for (std::size_t i = 0; i < assimpMesh->mNumFaces; ++i) {
        aiFace & face = assimpMesh->mFaces[i];
        if (face.mNumIndices != 3) {
            throw std::runtime_error("not a triangle mesh");
        }
        faces3[i] = make_uint3(face.mIndices[0],face.mIndices[1],face.mIndices[2]);
    }

    model->faces = malloc( assimpMesh->mNumFaces*sizeof(int)*3 );
    memcpy(model->faces, faces3.data(), assimpMesh->mNumFaces*sizeof(int)*3);

    aiReleaseImport(scene);
}


void Render::initializeBuffers(MyModel* model, std::string textureName, GLuint vertexbuffer, GLuint indexbuffer)
{
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, model->num_vertices*sizeof(float)*3, model->vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, model->num_faces*sizeof(int)*3, model->faces, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


// Camera Axis:
//   X - Right, Y - Down, Z - Forward
// Image Origin:
//   Top Left
// Pricipal point specified with image origin (0,0) at top left of top-left pixel (not center)
void Render::ProjectionMatrixRDF_TopLeft(float* m, int w, int h, float fu, float fv, float u0, float v0, float zNear, float zFar )
{
    // http://www.songho.ca/opengl/gl_projectionmatrix.html
    const float L = -(u0) * zNear / fu;
    const float R = +(w-u0) * zNear / fu;
    const float T = -(v0) * zNear / fv;
    const float B = +(h-v0) * zNear / fv;
    
    std::fill_n(m, 4*4, 0);
    
    m[0*4+0] = 2 * zNear / (R-L);
    m[1*4+1] = 2 * zNear / (T-B);
    
    m[2*4+0] = (R+L)/(L-R);
    m[2*4+1] = (T+B)/(B-T);
    m[2*4+2] = (zFar +zNear) / (zFar - zNear);
    m[2*4+3] = 1.0;
    
    m[3*4+2] =  (2*zFar*zNear)/(zNear - zFar);
}


void Render::print_matrix(float *m)
{
    for(int r=0; r< 4; ++r) {
        for(int c=0; c<4; ++c) {
            std::cout << m[4*c+r] << '\t';
        }
        std::cout << std::endl;
    }
}


float Render::render(const float* data, const int* labels, const float* rois, int num_rois, int num_gt, int num_classes, int width, int height,
                    const float* poses_gt, const float* poses_pred, const float* poses_init, float* bottom_diff, const float* meta_data, int num_meta_data)
{
  void *buffer;
  float projectionMatrix[16], mvMatrix[16];

  // build context
  OSMesaContext ctx = OSMesaCreateContextExt( OSMESA_RGB, 16, 0, 0, NULL );

  if (!ctx) 
  {
    printf("OSMesaCreateContext failed!\n");
    return 0;
  }

   /* Allocate the image buffer */
   buffer = malloc( width * height * 3 * sizeof(GLubyte) );
   if (!buffer) {
      printf("Alloc image buffer failed!\n");
      return 0;
   }

   /* Bind the buffer to the context and make it current */
   if (!OSMesaMakeCurrent( ctx, buffer, GL_UNSIGNED_BYTE, width, height )) {
      printf("OSMesaMakeCurrent failed!\n");
      return 0;
   }

  // initialize buffers
  GLuint vertexbuffer, indexbuffer;
  // Generate 1 buffer, put the resulting identifier in vertexbuffer
  glGenBuffers(1, &vertexbuffer);
  glGenBuffers(1, &indexbuffer);

  // show gt pose
  std::vector<cv::Mat*> gt_masks(num_gt);
  for (int n = 0; n < num_gt; n++)
  {
    int batch_id = int(poses_gt[n * 13 + 0]);
    float fx = meta_data[batch_id * num_meta_data + 0];
    float fy = meta_data[batch_id * num_meta_data + 4];
    float px = meta_data[batch_id * num_meta_data + 2];
    float py = meta_data[batch_id * num_meta_data + 5];

    ProjectionMatrixRDF_TopLeft(projectionMatrix, width, height, fx, -fy, px+0.5, height-(py+0.5), 0.25, 6.0);

    int class_id = int(poses_gt[n * 13 + 1]);

    initializeBuffers(models_[class_id-1], texture_names_[class_id-1], vertexbuffer, indexbuffer);
    Eigen::Quaterniond quaternion(poses_gt[n * 13 + 6], poses_gt[n * 13 + 7], poses_gt[n * 13 + 8], poses_gt[n * 13 + 9]);
    Sophus::SE3d::Point translation(poses_gt[n * 13 + 10], poses_gt[n * 13 + 11], poses_gt[n * 13 + 12]);
    const Sophus::SE3d T_co(quaternion, translation);

    // OpenGL rendering
    glColor3ub(255,255,255);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projectionMatrix);

    glMatrixMode(GL_MODELVIEW);
    Eigen::Matrix4f mv = T_co.cast<float>().matrix();
    OpenGlMatrix(mv, mvMatrix);
    glLoadMatrixf(mvMatrix);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
    glDrawElements(GL_TRIANGLES, models_[class_id-1]->num_faces * 3, GL_UNSIGNED_INT, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);

    glFinish();

    gt_masks[n] = new cv::Mat(height, width, CV_8UC3);
    memcpy( gt_masks[n]->data, buffer, width * height * 3 * sizeof(GLubyte) );

    // std::string filename = std::to_string(counter_++) + "_gt.ppm";
    // cv::imwrite(filename.c_str(), *(gt_masks[n]) );
    // write_ppm(filename.c_str(), (GLubyte*)buffer, width, height);
  }

  // show predicted pose
  cv::Mat dst1;
  cv::Mat dst2;
  double delta = 0.001;
  float loss = 0;
  for (int n = 0; n < num_rois; n++)
  {
    int batch_id = int(rois[n * 6 + 0]);
    float fx = meta_data[batch_id * num_meta_data + 0];
    float fy = meta_data[batch_id * num_meta_data + 4];
    float px = meta_data[batch_id * num_meta_data + 2];
    float py = meta_data[batch_id * num_meta_data + 5];

    ProjectionMatrixRDF_TopLeft(projectionMatrix, width, height, fx, -fy, px+0.5, height-(py+0.5), 0.25, 6.0);

    int class_id = int(rois[n * 6 + 1]);

    // find the gt index
    int gt_ind = -1;
    for (int i = 0; i < num_gt; i++)
    {
      int gt_batch = int(poses_gt[i * 13 + 0]);
      int gt_id = int(poses_gt[i * 13 + 1]);
      if(class_id == gt_id && batch_id == gt_batch)
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

    initializeBuffers(models_[class_id-1], texture_names_[class_id-1], vertexbuffer, indexbuffer);

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

      glColor3ub(255,255,255);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glMatrixMode(GL_PROJECTION);
      glLoadMatrixf(projectionMatrix);
      glMatrixMode(GL_MODELVIEW);

      Eigen::Matrix4f mv = T_co_pred.cast<float>().matrix();
      OpenGlMatrix(mv, mvMatrix);
      glLoadMatrixf(mvMatrix);

      glEnableClientState(GL_VERTEX_ARRAY);
      glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
      glVertexPointer(3, GL_FLOAT, 0, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexbuffer);
      glDrawElements(GL_TRIANGLES, models_[class_id-1]->num_faces * 3, GL_UNSIGNED_INT, 0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      glDisableClientState(GL_VERTEX_ARRAY);

      glFinish();

      cv::Mat mask(height, width, CV_8UC3);
      memcpy( mask.data, buffer, width * height * 3 * sizeof(GLubyte) );

      // compute the overlap between masks
      cv::bitwise_and(mask, *gt_masks[gt_ind], dst1);
      cv::bitwise_or(mask, *gt_masks[gt_ind], dst2);
      IoUs[i] = cv::sum(dst1)[0] / cv::sum(dst2)[0];

      // std::string filename = std::to_string(counter_++) + "_pose.ppm";
      // cv::imwrite(filename.c_str(), mask );

    }  // end rendering

    // compute loss and gradient
    loss += (1.0 - IoUs[0]) / num_rois;

    bottom_diff[n * 4 * num_classes + 4 * class_id + 0] = (IoUs[0] - IoUs[1]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 1] = (IoUs[0] - IoUs[2]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 2] = (IoUs[0] - IoUs[3]) / delta / num_rois;
    bottom_diff[n * 4 * num_classes + 4 * class_id + 3] = (IoUs[0] - IoUs[4]) / delta / num_rois;
  }

  for (int n = 0; n < num_gt; n++)
    delete gt_masks[n];

  OSMesaDestroyContext( ctx );

  /* free the image buffer */
  free( buffer );

  return loss;
}

void Render::write_ppm(const char *filename, const GLubyte *buffer, int width, int height)
{
   const int binary = 0;
   FILE *f = fopen( filename, "w" );
   if (f) {
      int i, x, y;
      const GLubyte *ptr = buffer;
      if (binary) {
         fprintf(f,"P6\n");
         fprintf(f,"# ppm-file created by osdemo.c\n");
         fprintf(f,"%i %i\n", width,height);
         fprintf(f,"255\n");
         fclose(f);
         f = fopen( filename, "ab" );  /* reopen in binary append mode */
         for (y=height-1; y>=0; y--) {
            for (x=0; x<width; x++) {
               i = (y*width + x) * 3;
               fputc(ptr[i], f);   /* write red */
               fputc(ptr[i+1], f); /* write green */
               fputc(ptr[i+2], f); /* write blue */
            }
         }
      }
      else {
         /*ASCII*/
         int counter = 0;
         fprintf(f,"P3\n");
         fprintf(f,"# ascii ppm file created by osdemo.c\n");
         fprintf(f,"%i %i\n", width, height);
         fprintf(f,"255\n");
         for (y=height-1; y>=0; y--) {
            for (x=0; x<width; x++) {
               i = (y*width + x) * 3;
               fprintf(f, " %3d %3d %3d", ptr[i], ptr[i+1], ptr[i+2]);
               counter++;
               if (counter % 5 == 0)
                  fprintf(f, "\n");
            }
         }
      }
      fclose(f);
   }
}

int main(int argc, char** argv) 
{
  Render render;
  render.setup(argv[1]);

  //while (!pangolin::ShouldQuit()) 
  render.render(NULL, NULL, NULL, 0, 0, 0, 480, 640, NULL, NULL, NULL, NULL, NULL, 0);
}
