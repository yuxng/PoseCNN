#pragma once

#include <assert.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <df/camera/linear.h>
#include <df/util/fileHelpers.h>

namespace df {

template <typename RenderType>
class GLRenderer {
public:

    GLRenderer(const int renderWidth, const int renderHeight, const std::string cameraModelName = "Linear");

    void render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers, const GLenum mode = GL_TRIANGLES);

    void render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers,
                pangolin::GlBuffer & indexBuffer, const GLenum mode = GL_TRIANGLES);

    void render(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                const std::vector<pangolin::GlBuffer> & indexBuffers,
                const GLenum mode = GL_TRIANGLES);

    void render(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                const std::vector<pangolin::GlBuffer*> & indexBuffers,
                const std::vector<Eigen::Matrix4f> & transforms,
                const GLenum mode = GL_TRIANGLES);

    inline const pangolin::GlTextureCudaArray & texture(const int i) const {
        assert(i < RenderType::numTextures);
        return textures_[i];
    }

    void setProjectionMatrix(const Eigen::Matrix4f & m) { projectionMatrix_ = m; }

    void setModelViewMatrix(const Eigen::Matrix4f & m) { modelViewMatrix_ = m; }

    template <typename Scalar>
    void setCameraParams(const Scalar * params, const std::size_t nParams) {
        cameraParams_.resize(nParams);
        for (std::size_t i = 0; i < nParams; ++i) {
            cameraParams_[i] = params[i];
            std::cout << cameraParams_[i] << " ";
        } std::cout << std::endl;

        // focal length adjustment
        cameraParams_[0] /= (renderWidth_ / 2.f);
        cameraParams_[1] /= (renderHeight_ / 2.f);

        // principal point adjustment
        cameraParams_[2] = (cameraParams_[2] + 0.5) / (renderWidth_ / 2.f) - 1.f;
        cameraParams_[3] = (cameraParams_[3] + 0.5) / (renderHeight_ / 2.f) - 1.f;

    }

private:

    void renderSetup();

    void matrixSetup();

    void vertexAttributeSetup(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers);

    void renderTeardown(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers);

    int renderWidth_;
    int renderHeight_;

    pangolin::GlTextureCudaArray textures_[RenderType::numTextures];
    pangolin::GlRenderBuffer renderBuffer_;
    pangolin::GlFramebuffer frameBuffer_;

    pangolin::GlSlProgram program_;
    GLint projectionMatrixHandle_;
    GLint modelViewMatrixHandle_;
    GLint cameraParamsHandle_;

    Eigen::Matrix4f projectionMatrix_;
    Eigen::Matrix4f modelViewMatrix_;

    std::vector<float> cameraParams_;

};

// -=-=-=-=-=-=-=-=- implementation -=-=-=-=-=-=-=-=-
template <typename RenderType>
GLRenderer<RenderType>::GLRenderer(const int renderWidth, const int renderHeight,
                                   const std::string cameraModelName)
    : renderWidth_(renderWidth), renderHeight_(renderHeight) {

    // initialize the frame buffer
    for (int i = 0; i < RenderType::numTextures; ++i) {

        textures_[i].Reinitialise(renderWidth,renderHeight,RenderType::textureFormats()[i]);
        frameBuffer_.AttachColour(textures_[i]);

    }

    renderBuffer_.Reinitialise(renderWidth,renderHeight);
    frameBuffer_.AttachDepth(renderBuffer_);

    // initialize the GlSl program
    const std::string shaderDir = compileDirectory() + "/shaders/";

    std::string vertShaderFile = shaderDir + RenderType::vertShaderName();
    while (vertShaderFile.find_first_of('#') < vertShaderFile.length() ) {

        vertShaderFile.replace(vertShaderFile.find('#'), 1, cameraModelName);

    }

    std::cout << "vert shader: " << vertShaderFile << std::endl;

    const std::string fragShaderFile = shaderDir + RenderType::fragShaderName();

    program_.AddShaderFromFile(pangolin::GlSlVertexShader, vertShaderFile);
    program_.AddShaderFromFile(pangolin::GlSlFragmentShader, fragShaderFile);
    program_.Link();

    projectionMatrixHandle_ = program_.GetUniformHandle("projectionMatrix");
    modelViewMatrixHandle_ = program_.GetUniformHandle("modelViewMatrix");
    cameraParamsHandle_ = program_.GetUniformHandle("cameraParams");

    std::cout << "projection handle: " << projectionMatrixHandle_ << std::endl;

    std::cout << "modelview matrix handle: " << modelViewMatrixHandle_ << std::endl;

    std::cout << "camera param handle: " << cameraParamsHandle_ << std::endl;

}

template <typename RenderType>
void GLRenderer<RenderType>::renderSetup() {

    frameBuffer_.Bind();

    glEnable(GL_SCISSOR_TEST);
    glViewport(0,0,renderWidth_,renderHeight_);
    glScissor(0,0,renderWidth_,renderHeight_);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    program_.Bind();

}

template <typename RenderType>
void GLRenderer<RenderType>::matrixSetup() {

    if (projectionMatrixHandle_ >= 0) {
        glUniformMatrix4fv(projectionMatrixHandle_, 1, GL_FALSE, projectionMatrix_.data());
    }
    if (cameraParamsHandle_ >= 0) {
        glUniform1fv(cameraParamsHandle_, cameraParams_.size(), cameraParams_.data());
    }
    glUniformMatrix4fv(modelViewMatrixHandle_, 1, GL_FALSE, modelViewMatrix_.data());

}

template <typename RenderType>
void GLRenderer<RenderType>::vertexAttributeSetup(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers) {

    static constexpr int numAttributes = RenderType::numVertexAttributes;
    assert(vertexAttributeBuffers.size() == numAttributes);

    for (uint i = 0; i < numAttributes; ++i) {

        const uint size = RenderType::vertexAttributeSizes()[i];
        const GLenum type = RenderType::vertexAttributeTypes()[i];

        assert(size == vertexAttributeBuffers[i]->count_per_element);
        assert(type == vertexAttributeBuffers[i]->datatype);

        vertexAttributeBuffers[i]->Bind();
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, size, type, GL_FALSE, 0, 0);

    }

}

template <typename RenderType>
void GLRenderer<RenderType>::renderTeardown(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers) {

    static constexpr int numAttributes = RenderType::numVertexAttributes;

    for (uint i = 0; i < numAttributes; ++i) {
        vertexAttributeBuffers[i]->Unbind();
        glDisableVertexAttribArray(i);
    }

    program_.Unbind();

    frameBuffer_.Unbind();

}

template <typename RenderType>
void GLRenderer<RenderType>::render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers, const GLenum mode) {

    renderSetup();

    matrixSetup();

    vertexAttributeSetup(vertexAttributeBuffers);

    const int N = vertexAttributeBuffers[0]->num_elements;
    glDrawArrays(mode, 0, N);

    renderTeardown(vertexAttributeBuffers);

}

template <typename RenderType>
void GLRenderer<RenderType>::render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers,
                                    pangolin::GlBuffer & indexBuffer,
                                    const GLenum mode) {

    renderSetup();

    matrixSetup();

    vertexAttributeSetup(vertexAttributeBuffers);

    indexBuffer.Bind();
    glDrawElements(mode, indexBuffer.num_elements, GL_UNSIGNED_INT, 0);

    indexBuffer.Unbind();

    renderTeardown(vertexAttributeBuffers);

}

template <typename RenderType>
void GLRenderer<RenderType>::render(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                                    const std::vector<pangolin::GlBuffer> & indexBuffers,
                                    const GLenum mode) {

    assert(indexBuffers.size() == vertexAttributeBuffers.size());

    renderSetup();

    matrixSetup();

    for (int m = 0; m < vertexAttributeBuffers.size(); ++m) {

        vertexAttributeSetup(vertexAttributeBuffers[m]);

        indexBuffers[m].Bind();

        glDrawElements(mode, indexBuffers[m].num_elements, GL_UNSIGNED_INT, 0);

        indexBuffers[m].Unbind();

    }

    renderTeardown(vertexAttributeBuffers.back());

}

template <typename RenderType>
void GLRenderer<RenderType>::render(const std::vector<std::vector<pangolin::GlBuffer *> > & vertexAttributeBuffers,
                                    const std::vector<pangolin::GlBuffer*> & indexBuffers,
                                    const std::vector<Eigen::Matrix4f> & transforms,
                                    const GLenum mode) {

    assert(indexBuffers.size() == vertexAttributeBuffers.size());

    renderSetup();

    for (int m = 0; m < vertexAttributeBuffers.size(); ++m) {

        setModelViewMatrix(transforms[m]);

        matrixSetup();

        vertexAttributeSetup(vertexAttributeBuffers[m]);

        indexBuffers[m]->Bind();

        glDrawElements(mode, indexBuffers[m]->num_elements, GL_UNSIGNED_INT, 0);

        indexBuffers[m]->Unbind();

    }

    renderTeardown(vertexAttributeBuffers.back());

}


} // namespace df
