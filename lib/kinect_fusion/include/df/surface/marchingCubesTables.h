#pragma once

#include <cuda_runtime.h>

namespace df {

typedef unsigned int VertexCountT;

extern VertexCountT vertexCountByVoxelCodeTable[256];

typedef unsigned int VertexIndexT;

extern VertexIndexT vertexIndicesByVoxelCodeTable[256][16];

} // namespace df
