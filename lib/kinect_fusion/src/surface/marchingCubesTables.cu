#include <df/surface/marchingCubesTables.h>
#include <df/util/cudaHelpers.h>

namespace df {

//texture<VertexCountT, 1, cudaReadModeElementType> vertexCountByVoxelCodeTex;
//texture<VertexCountT, 1, cudaReadModeElementType> vertexIndicesByVoxelCodeTex;

//class MarchingCubesMemoryManager {
//public:

//    MarchingCubesMemoryManager() {

//        cudaMalloc(&vertexCountData_, 256*sizeof(VertexCountT));
//        cudaMemcpy(vertexCountData_, vertexCountByVoxelCodeTable, 256*sizeof(VertexCountT), cudaMemcpyHostToDevice);

////        cudaChannelFormatDesc vertexCountChannelDesc = cudaCreateChannelDesc(8*sizeof(VertexCountT), 0, 0, 0, cudaChannelFormatKindUnsigned);
//        cudaBindTexture(0, vertexCountByVoxelCodeTex, vertexCountData_, 256*sizeof(VertexCountT)); //vertexCountChannelDesc);


//        cudaMalloc(&vertexIndexData_, 256*16*sizeof(VertexIndexT));
//        cudaMemcpy(vertexIndexData_, vertexIndicesByVoxelCodeTable, 256*16*sizeof(VertexIndexT), cudaMemcpyHostToDevice);
//        cudaBindTexture(0, vertexIndicesByVoxelCodeTex, vertexIndexData_, 256*16*sizeof(VertexIndexT));

//        cudaDeviceSynchronize();
//        CheckCudaDieOnError();

//    }

//    ~MarchingCubesMemoryManager() {

//        cudaUnbindTexture(vertexCountByVoxelCodeTex);
//        cudaFree(vertexCountData_);

//        cudaUnbindTexture(vertexIndicesByVoxelCodeTex);
//        cudaFree(vertexIndexData_);

//    }

//private:

//    VertexCountT * vertexCountData_;

//    VertexIndexT * vertexIndexData_;

//};

//void initMarchingCubesTables() {
//    static MarchingCubesMemoryManager manager;
//}

} // namespace df
