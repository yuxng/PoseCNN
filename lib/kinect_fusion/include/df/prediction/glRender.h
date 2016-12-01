#pragma once

#include <assert.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <df/util/fileHelpers.h>

namespace df {

template <typename RenderType>
class GLRenderer {
public:

    GLRenderer(const int renderWidth, const int renderHeight);

    void render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers, const GLenum mode = GL_TRIANGLES);

    void render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers,
                pangolin::GlBuffer & indexBuffer, const GLenum mode = GL_TRIANGLES);

    inline const pangolin::GlTextureCudaArray & texture(const int i) const {
        assert(i < RenderType::numTextures);
        return textures_[i];
    }

    void setProjectionMatrix(const Eigen::Matrix4f & m) { projectionMatrix_ = m; }

    void setModelViewMatrix(const Eigen::Matrix4f & m) { modelViewMatrix_ = m; }


private:

    void renderSetup(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers);

    void renderTeardown(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers);

    int renderWidth_;
    int renderHeight_;

    pangolin::GlTextureCudaArray textures_[RenderType::numTextures];
    pangolin::GlRenderBuffer renderBuffer_;
    pangolin::GlFramebuffer frameBuffer_;

    pangolin::GlSlProgram program_;
    GLint projectionMatrixHandle_;
    GLint modelViewMatrixHandle_;

    Eigen::Matrix4f projectionMatrix_;
    Eigen::Matrix4f modelViewMatrix_;

};

// -=-=-=-=-=-=-=-=- implementation -=-=-=-=-=-=-=-=-
template <typename RenderType>
GLRenderer<RenderType>::GLRenderer(const int renderWidth, const int renderHeight)
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
    const std::string vertShaderFile = shaderDir + RenderType::vertShaderName();
    const std::string fragShaderFile = shaderDir + RenderType::fragShaderName();

    program_.AddShaderFromFile(pangolin::GlSlVertexShader, vertShaderFile);
    program_.AddShaderFromFile(pangolin::GlSlFragmentShader, fragShaderFile);
    program_.Link();

    projectionMatrixHandle_ = program_.GetUniformHandle("projectionMatrix");
    modelViewMatrixHandle_ = program_.GetUniformHandle("modelViewMatrix");

}

template <typename RenderType>
void GLRenderer<RenderType>::renderSetup(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers) {

    frameBuffer_.Bind();

    glEnable(GL_SCISSOR_TEST);
    glViewport(0,0,renderWidth_,renderHeight_);
    glScissor(0,0,renderWidth_,renderHeight_);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    program_.Bind();

    glUniformMatrix4fv(projectionMatrixHandle_, 1, GL_FALSE, projectionMatrix_.data());
    glUniformMatrix4fv(modelViewMatrixHandle_, 1, GL_FALSE, modelViewMatrix_.data());

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

    renderSetup(vertexAttributeBuffers);

    const int N = vertexAttributeBuffers[0]->num_elements;
    glDrawArrays(mode, 0, N);

    renderTeardown(vertexAttributeBuffers);

}

template <typename RenderType>
void GLRenderer<RenderType>::render(const std::vector<pangolin::GlBuffer *> & vertexAttributeBuffers,
                                    pangolin::GlBuffer & indexBuffer,
                                    const GLenum mode) {

    renderSetup(vertexAttributeBuffers);

    indexBuffer.Bind();
    glDrawElements(mode, indexBuffer.num_elements, GL_UNSIGNED_INT, 0);

    indexBuffer.Unbind();

    renderTeardown(vertexAttributeBuffers);

}

} // namespace df
