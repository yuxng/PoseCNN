#pragma once

#include <string>

namespace df {

struct VertAndNormalRenderType {
    static std::string vertShaderName() {
        static const char name[] = "vertsAndNorms.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "vertsAndNorms.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 2;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F, GL_RGBA32F };
        return formats;
    }
    static constexpr int numVertexAttributes = 2;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

struct VertNormalAndColorRenderType {
    static std::string vertShaderName() {
        static const char name[] = "vertsNormsAndColor.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "vertsNormsAndColor.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 3;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F, GL_RGBA32F, GL_RGBA32F };
        return formats;
    }
    static constexpr int numVertexAttributes = 3;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT, GL_FLOAT };
        return types;
    }
};

struct CanonicalVertRenderType {
    static std::string vertShaderName() {
        static const char name[] = "canonicalVerts.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "canonicalVerts.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 1;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F };
        return formats;
    }
    static constexpr int numVertexAttributes = 2;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT };
        return types;
    }
};

struct CanonicalVertAndNormalRenderType {
    static std::string vertShaderName() {
        static const char name[] = "canonicalVertsAndNorms.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "canonicalVertsAndNorms.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 2;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F, GL_RGBA32F };
        return formats;
    }
    static constexpr int numVertexAttributes = 3;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT, GL_FLOAT };
        return types;
    }
};


struct CanonicalVertAndTextureRenderType {
    static std::string vertShaderName() {
        static const char name[] = "canonicalVertsAndTexture.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "canonicalVertsAndTexture.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 3;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F, GL_RGBA32F, GL_RGBA32F};
        return formats;
    }
    static constexpr int numVertexAttributes = 4;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3, 2, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT };
        return types;
    }
};


struct CanonicalVertAndColorRenderType {
    static std::string vertShaderName() {
        static const char name[] = "canonicalVertsAndColor.vert";
        return std::string(name);
    }
    static std::string fragShaderName() {
        static const char name[] = "canonicalVertsAndColor.frag";
        return std::string(name);
    }
    static constexpr int numTextures = 3;
    static const GLenum * textureFormats() {
        static const GLenum formats[numTextures] = { GL_RGBA32F, GL_RGBA32F, GL_RGBA32F};
        return formats;
    }
    static constexpr int numVertexAttributes = 4;
    static const int * vertexAttributeSizes() {
        static const int sizes[numVertexAttributes] = { 3, 3, 3, 3 };
        return sizes;
    }
    static const GLenum * vertexAttributeTypes() {
        static const GLenum types[numVertexAttributes] = { GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT };
        return types;
    }
};

} // namespace df
