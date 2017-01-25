#include <df/surface/marchingCubes.h>

#include <df/camera/poly3.h>
#include <df/surface/marchingCubesTables.h>
#include <df/util/cudaHelpers.h>
#include <df/util/eigenHelpers.h>
#include <df/voxel/color.h>
#include <df/voxel/probability.h>
#include <df/voxel/compositeVoxel.h>
#include <df/voxel/tsdf.h>
#include <df/transform/rigid.h>

#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace df {

//extern texture<VertexCountT, 1, cudaReadModeElementType> vertexCountByVoxelCodeTex;
//extern texture<VertexIndexT, 1, cudaReadModeElementType> vertexIndicesByVoxelCodeTex;

texture<VertexCountT, 1, cudaReadModeElementType> vertexCountByVoxelCodeTex;
texture<VertexCountT, 1, cudaReadModeElementType> vertexIndicesByVoxelCodeTex;

class MarchingCubesMemoryManager {
public:

    MarchingCubesMemoryManager() {

        cudaMalloc(&vertexCountData_, 256*sizeof(VertexCountT));
        cudaMemcpy(vertexCountData_, vertexCountByVoxelCodeTable, 256*sizeof(VertexCountT), cudaMemcpyHostToDevice);

//        cudaChannelFormatDesc vertexCountChannelDesc = cudaCreateChannelDesc(8*sizeof(VertexCountT), 0, 0, 0, cudaChannelFormatKindUnsigned);
        cudaBindTexture(0, vertexCountByVoxelCodeTex, vertexCountData_, 256*sizeof(VertexCountT)); //vertexCountChannelDesc);

        cudaMalloc(&vertexIndexData_, 256*16*sizeof(VertexIndexT));
        cudaMemcpy(vertexIndexData_, vertexIndicesByVoxelCodeTable, 256*16*sizeof(VertexIndexT), cudaMemcpyHostToDevice);
        cudaBindTexture(0, vertexIndicesByVoxelCodeTex, vertexIndexData_, 256*16*sizeof(VertexIndexT));

        cudaDeviceSynchronize();
        CheckCudaDieOnError();

    }

    ~MarchingCubesMemoryManager() {

        cudaUnbindTexture(vertexCountByVoxelCodeTex);
        cudaFree(vertexCountData_);

        cudaUnbindTexture(vertexIndicesByVoxelCodeTex);
        cudaFree(vertexIndexData_);

    }

private:

    VertexCountT * vertexCountData_;

    VertexIndexT * vertexIndexData_;

};

void initMarchingCubesTables() {
    static MarchingCubesMemoryManager manager;
}



template <typename Scalar,
          typename VoxelT>
inline __device__ Scalar sampleVoxelGrid(const Tensor<3,VoxelT,DeviceResident> voxelGrid,
                                         const int x, const int y, const int z,
                                         const Scalar weightThreshold,
                                         bool & missingData) {
    const VoxelT & voxel = voxelGrid(x,y,z);
    if (voxel.template weight<TsdfVoxel>() >= weightThreshold) {
        return voxel.template value<TsdfVoxel>();
    }
    else {
        missingData = true;
        return Scalar(0);
    }

}

template <typename VoxelT>
inline __device__ auto sampleVoxelGrid(const Tensor<3,VoxelT,DeviceResident> voxelGrid,
                                       const int x, const int y, const int z) -> decltype(voxelGrid(0,0,0).template value<TsdfVoxel>()) {

    const VoxelT & voxel = voxelGrid(x,y,z);
    return voxel.template value<TsdfVoxel>();

}

//
template <typename Scalar,
          typename VoxelT>
__global__ void classifyVoxelsKernel(const Tensor<3,VoxelT,DeviceResident> voxelGrid,
                                     const Scalar weightThreshold,
                                     //Tensor<3,uint,DeviceResident> voxelCodes, // TODO: experiment with data size
                                     Tensor<3,uint,DeviceResident> vertexCounts) {

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = threadIdx.y + blockDim.y * blockIdx.y;
    const uint z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < voxelGrid.dimensionSize(0)) && (y < voxelGrid.dimensionSize(1)) && (z < voxelGrid.dimensionSize(2))) {

        uint voxelCode, numVertices;

        if ((x == (voxelGrid.dimensionSize(0) - 1)) || (y == (voxelGrid.dimensionSize(1) - 1)) || (z == (voxelGrid.dimensionSize(2) - 1))) {

            // cannot do binlinear interpolation on these vertices
            voxelCode = 0;
            numVertices = 0;

        } else {

            bool missingData = false;
            Scalar centerVals[8];
            centerVals[0] = sampleVoxelGrid(voxelGrid, x,     y,     z,     weightThreshold, missingData);
            centerVals[1] = sampleVoxelGrid(voxelGrid, x + 1, y,     z,     weightThreshold, missingData);
            centerVals[2] = sampleVoxelGrid(voxelGrid, x + 1, y + 1, z,     weightThreshold, missingData);
            centerVals[3] = sampleVoxelGrid(voxelGrid, x,     y + 1, z,     weightThreshold, missingData);
            centerVals[4] = sampleVoxelGrid(voxelGrid, x,     y,     z + 1, weightThreshold, missingData);
            centerVals[5] = sampleVoxelGrid(voxelGrid, x + 1, y,     z + 1, weightThreshold, missingData);
            centerVals[6] = sampleVoxelGrid(voxelGrid, x + 1, y + 1, z + 1, weightThreshold, missingData);
            centerVals[7] = sampleVoxelGrid(voxelGrid, x,     y + 1, z + 1, weightThreshold, missingData);

            if (missingData) {

                voxelCode = 0;
                numVertices = 0;

            } else {

    //            printf("8 valid\n");

                voxelCode  = uint(centerVals[0] < Scalar(0));
                voxelCode += uint(centerVals[1] < Scalar(0)) << 1;
                voxelCode += uint(centerVals[2] < Scalar(0)) << 2;
                voxelCode += uint(centerVals[3] < Scalar(0)) << 3;
                voxelCode += uint(centerVals[4] < Scalar(0)) << 4;
                voxelCode += uint(centerVals[5] < Scalar(0)) << 5;
                voxelCode += uint(centerVals[6] < Scalar(0)) << 6;
                voxelCode += uint(centerVals[7] < Scalar(0)) << 7;

    //            printf("vertex code %d\n",voxelCode);

                // TODO: try constant memory as well
                numVertices = tex1Dfetch(vertexCountByVoxelCodeTex, voxelCode);

            }

        }

//        voxelCodes(x,y,z) = voxelCode;
        vertexCounts(x,y,z) = numVertices;


//        if (numVertices > 0) {
//            atomicAdd(validVoxelCount,1);
//            printf("%d\n",numVertices);
//        }

    }
}


struct Binarizer {
    inline __host__ __device__ uint operator()(const uint & val) { return val > 0 ? 1 : 0; }
};


__global__ void computeValidVoxelIndicesKernel(const Tensor<3,uint,DeviceResident> vertexCounts,
                                               const Tensor<3,uint,DeviceResident> validVoxelScanResult,
                                               Tensor<1,uint,DeviceResident> validVoxelIndices) {

    const uint x = threadIdx.x + blockDim.x * blockIdx.x;
    const uint y = threadIdx.y + blockDim.y * blockIdx.y;
    const uint z = threadIdx.z + blockDim.z * blockIdx.z;

//    const uint i = threadIdx.x + blockDim.x * blockIdx.x;

//    if (threadIdx.x == 0) {
//        printf("%d m\n",blockIdx.x);
//    }

    if ( (x < vertexCounts.dimensionSize(0)) && (y < vertexCounts.dimensionSize(1)) && (z < vertexCounts.dimensionSize(2))) {

        if (vertexCounts(x,y,z) > 0) {

//            printf("%d,%d,%d valid \n",x,y,z);
//            atomicAdd(nValid,1);

            const uint i = x + vertexCounts.dimensionSize(0)*(y + vertexCounts.dimensionSize(1)*z);

            const uint compactedIndex = validVoxelScanResult(x,y,z);

//            if (compactedIndex < 50) {
//                printf("%d (%d,%d,%d) valid -> %d\n",i,x,y,z,compactedIndex);
//            }

            validVoxelIndices(compactedIndex) = i;

        }

    }

}

template <typename Scalar>
inline __device__ Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> computeVertex(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & voxelCenterA,
                                                                           const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & voxelCenterB,
                                                                           const Scalar valueA, const Scalar valueB) {

    const Scalar t = ( -valueA ) / ( valueB - valueA );
    return voxelCenterA + t*(voxelCenterB - voxelCenterA);

}

template <typename Scalar, typename VoxelT>
__global__ void computeTrianglesKernel(const Tensor<1,uint,DeviceResident> validVoxelIndices,
                                       const Tensor<3,uint,DeviceResident> vertexCountScanResult,
                                       const Tensor<3,VoxelT,DeviceResident> voxelGrid,
                                       //const Tensor<3,uint,DeviceResident> voxelCodes,
                                       Tensor<2,Scalar,DeviceResident> vertices) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    const uint i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < validVoxelIndices.dimensionSize(0)) {

        const uint index = validVoxelIndices(i);

//        if (vertexCountScanResult.data()[index] < 5) {
//            printf("%d: %d !!\n",index,vertexCountScanResult.data()[index]);
//        }

//        printf("%d: %d\n",i,index);

        const uint x = index % voxelGrid.dimensionSize(0);
        const uint y = (index / voxelGrid.dimensionSize(0)) % voxelGrid.dimensionSize(1);
        const uint z = index / (voxelGrid.dimensionSize(0)*voxelGrid.dimensionSize(1));


//        if (vertexCountScanResult.data()[index] < 5) {
//            printf("%d: %d %d %d $\n",index,x,y,z);
//        }

        Vec3 vertexCenters[8];
        vertexCenters[0] = Vec3(x,     y,     z    );
        vertexCenters[1] = Vec3(x + 1, y,     z    );
        vertexCenters[2] = Vec3(x + 1, y + 1, z    );
        vertexCenters[3] = Vec3(x,     y + 1, z    );
        vertexCenters[4] = Vec3(x,     y,     z + 1);
        vertexCenters[5] = Vec3(x + 1, y,     z + 1);
        vertexCenters[6] = Vec3(x + 1, y + 1, z + 1);
        vertexCenters[7] = Vec3(x,     y + 1, z + 1);

        Scalar centerVals[8];
        centerVals[0] = sampleVoxelGrid(voxelGrid, x,     y,     z   );
        centerVals[1] = sampleVoxelGrid(voxelGrid, x + 1, y,     z   );
        centerVals[2] = sampleVoxelGrid(voxelGrid, x + 1, y + 1, z   );
        centerVals[3] = sampleVoxelGrid(voxelGrid, x,     y + 1, z   );
        centerVals[4] = sampleVoxelGrid(voxelGrid, x,     y,     z + 1);
        centerVals[5] = sampleVoxelGrid(voxelGrid, x + 1, y,     z + 1);
        centerVals[6] = sampleVoxelGrid(voxelGrid, x + 1, y + 1, z + 1);
        centerVals[7] = sampleVoxelGrid(voxelGrid, x,     y + 1, z + 1);

        static constexpr int maxVertsPerVoxel = 12;
        // TODO: Richard's code uses 32 --- why? there's enough memory for up to 256
        static constexpr int numThreads = 256;

        // TODO: make dynamic? will there be a performance hit?
        __shared__ char s[maxVertsPerVoxel*numThreads*sizeof(Vec3)];
        // avoids constructor issues
        Vec3 * potentialVertexList = reinterpret_cast<Vec3 *>(&s[0]);

        // TODO: why strided like this? is it faster the other way?
        // TODO: use a fancy dispatch mechanism to compute only necessary verts?
        potentialVertexList[threadIdx.x +  0*numThreads] = computeVertex(vertexCenters[0],vertexCenters[1],centerVals[0],centerVals[1]);
        potentialVertexList[threadIdx.x +  1*numThreads] = computeVertex(vertexCenters[1],vertexCenters[2],centerVals[1],centerVals[2]);
        potentialVertexList[threadIdx.x +  2*numThreads] = computeVertex(vertexCenters[2],vertexCenters[3],centerVals[2],centerVals[3]);
        potentialVertexList[threadIdx.x +  3*numThreads] = computeVertex(vertexCenters[3],vertexCenters[0],centerVals[3],centerVals[0]);
        potentialVertexList[threadIdx.x +  4*numThreads] = computeVertex(vertexCenters[4],vertexCenters[5],centerVals[4],centerVals[5]);
        potentialVertexList[threadIdx.x +  5*numThreads] = computeVertex(vertexCenters[5],vertexCenters[6],centerVals[5],centerVals[6]);
        potentialVertexList[threadIdx.x +  6*numThreads] = computeVertex(vertexCenters[6],vertexCenters[7],centerVals[6],centerVals[7]);
        potentialVertexList[threadIdx.x +  7*numThreads] = computeVertex(vertexCenters[7],vertexCenters[4],centerVals[7],centerVals[4]);
        potentialVertexList[threadIdx.x +  8*numThreads] = computeVertex(vertexCenters[0],vertexCenters[4],centerVals[0],centerVals[4]);
        potentialVertexList[threadIdx.x +  9*numThreads] = computeVertex(vertexCenters[1],vertexCenters[5],centerVals[1],centerVals[5]);
        potentialVertexList[threadIdx.x + 10*numThreads] = computeVertex(vertexCenters[2],vertexCenters[6],centerVals[2],centerVals[6]);
        potentialVertexList[threadIdx.x + 11*numThreads] = computeVertex(vertexCenters[3],vertexCenters[7],centerVals[3],centerVals[7]);
        __syncthreads();

        // TODO: recompute?
        //const uint voxelCode = voxelCodes(x,y,z);
        uint voxelCode;
        voxelCode  = uint(centerVals[0] < Scalar(0));
        voxelCode += uint(centerVals[1] < Scalar(0)) << 1;
        voxelCode += uint(centerVals[2] < Scalar(0)) << 2;
        voxelCode += uint(centerVals[3] < Scalar(0)) << 3;
        voxelCode += uint(centerVals[4] < Scalar(0)) << 4;
        voxelCode += uint(centerVals[5] < Scalar(0)) << 5;
        voxelCode += uint(centerVals[6] < Scalar(0)) << 6;
        voxelCode += uint(centerVals[7] < Scalar(0)) << 7;

        const uint numVertices = tex1Dfetch(vertexCountByVoxelCodeTex, voxelCode);


//        if (vertexCountScanResult.data()[index] < 5) {
//            printf("%d: %d ?!\n",index,voxelCode);
//            printf("%d: %d ??\n",index,numVertices);
//        }

        for (uint v = 0; v < numVertices; ++v) {

            const uint vertexIndex = tex1Dfetch(vertexIndicesByVoxelCodeTex, voxelCode*16 + v);

            const uint outputIndex = vertexCountScanResult.data()[index] + v;

            Eigen::Map<Vec3> map(&vertices(0,outputIndex));
            map = potentialVertexList[threadIdx.x + vertexIndex*numThreads];

//            if (outputIndex < 10) {
//                printf("%d: %f %f %f\n",outputIndex,map(0),map(1),map(2));
//            }

        }

    }


}

template <typename Scalar,
          typename VoxelT>
void extractSurface(ManagedTensor<2, Scalar, DeviceResident> & vertices,
                    const VoxelGrid<Scalar,VoxelT,DeviceResident> & voxelGrid,
                    const Scalar weightThreshold) {

    std::cout << "threshold: " << weightThreshold << std::endl;

    // TODO: ideas to make this faster
    //
    // 1. the extra storage for whether or not each voxel contains geometry is probably wasteful
    // and only done so we can use thrust::exclusive_scan. this could perhaps be made faster
    // with a custom implementation of the exclusive_scan
    //
    // 2. Richard's code claims that recalculating the voxel code is faster than storing it in
    // global memory. this should also be investigated

//    ManagedTensor<3,uint,DeviceResident> voxelCodes(voxelGrid.dimensions());
    static ManagedTensor<3,uint,DeviceResident> dVertexCounts(voxelGrid.dimensions());

//    int * dValidVoxelCount;
//    cudaMalloc(&dValidVoxelCount,sizeof(int));
//    cudaMemset(dValidVoxelCount,0,sizeof(int));

    {
        dim3 block(16,16,4);
        dim3 grid(voxelGrid.size(0)/block.x,voxelGrid.size(1)/block.y,voxelGrid.size(2)/block.z);

        classifyVoxelsKernel<<<grid,block>>>(voxelGrid.grid(),weightThreshold,
//                                             voxelCodes,
                                             dVertexCounts);
    }

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

    static ManagedTensor<3,uint,DeviceResident> vertexCountScanResult(voxelGrid.dimensions());

    thrust::exclusive_scan(thrust::device_ptr<uint>(dVertexCounts.data()),
                           thrust::device_ptr<uint>(dVertexCounts.data() + dVertexCounts.count()),
                           thrust::device_ptr<uint>(vertexCountScanResult.data()));

    uint numVertices;
    cudaMemcpy(&numVertices,vertexCountScanResult.data() + vertexCountScanResult.count()-1,sizeof(uint),cudaMemcpyDeviceToHost);
    uint lastNumVertices;
    cudaMemcpy(&lastNumVertices,dVertexCounts.data() + dVertexCounts.count()-1,sizeof(uint),cudaMemcpyDeviceToHost);
    numVertices += lastNumVertices;
    printf("%d vertices\n",numVertices);

    static ManagedTensor<3,uint,DeviceResident> validVoxelScanResult(voxelGrid.dimensions());
    thrust::transform(thrust::device_ptr<uint>(dVertexCounts.data()),
                      thrust::device_ptr<uint>(dVertexCounts.data() + dVertexCounts.count()),
                      thrust::device_ptr<uint>(validVoxelScanResult.data()),
                      Binarizer());

    thrust::exclusive_scan(thrust::device_ptr<uint>(validVoxelScanResult.data()),
                           thrust::device_ptr<uint>(validVoxelScanResult.data() + validVoxelScanResult.count()),
                           thrust::device_ptr<uint>(validVoxelScanResult.data()));

    uint numValidVoxels;
    cudaMemcpy(&numValidVoxels,validVoxelScanResult.data() + validVoxelScanResult.count()-1,sizeof(uint),cudaMemcpyDeviceToHost);
    if (lastNumVertices > 0) {
        ++numValidVoxels;
    }
//    printf("%d valid voxels\n",numValidVoxels);

//    cudaMemcpy(&numValidVoxels,dValidVoxelCount,sizeof(int),cudaMemcpyDeviceToHost);
//    printf("%d valid voxels\n",numValidVoxels);

//    cudaFree(dValidVoxelCount);

    Eigen::Matrix<uint,1,1> validVoxelIndicesDim(numValidVoxels);
    ManagedTensor<1,uint,DeviceResident> validVoxelIndices(validVoxelIndicesDim);

    {
//        const uint nThreads = 1024;
//        const uint nVoxels = voxelGrid.grid().count();
//        const uint grid = intDivideAndCeil(nVoxels,nThreads);

        dim3 block(16,16,4);
        dim3 grid(intDivideAndCeil(voxelGrid.size(0),block.x),
                  intDivideAndCeil(voxelGrid.size(1),block.y),
                  intDivideAndCeil(voxelGrid.size(2),block.z));

//        std::cout << "grid: " << grid << std::endl;

        computeValidVoxelIndicesKernel<<<grid,block>>>(dVertexCounts,
                                                       validVoxelScanResult,
                                                       validVoxelIndices);

    }


    cudaDeviceSynchronize();
    CheckCudaDieOnError();

    Eigen::Matrix<uint,2,1> verticesDim(3,numVertices);
    vertices.resize(verticesDim);

    {
        const uint nThreads = 256;
        computeTrianglesKernel<<<intDivideAndCeil(numValidVoxels,nThreads),nThreads>>>(validVoxelIndices,
                                                                                       vertexCountScanResult,
                                                                                       voxelGrid.grid(),
//                                                                                       voxelCodes,
                                                                                       vertices);
    }

    cudaDeviceSynchronize();
    CheckCudaDieOnError();

}



template <typename Scalar>
uint weldVertices(const Tensor<2,Scalar,DeviceResident> & vertices,
                  Tensor<2,Scalar,DeviceResident> & weldedVertices,
                  ManagedTensor<1,int,DeviceResident> & indices) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    assert(vertices.dimensionSize(0) == 3);
    const uint numVertices = vertices.dimensionSize(1);

    weldedVertices.copyFrom(vertices);

    thrust::device_ptr<Vec3> weldedVertexPointer(reinterpret_cast<Vec3 *>(weldedVertices.data()));

    thrust::sort( weldedVertexPointer, weldedVertexPointer + numVertices, VecLess<Scalar,3>() );

    thrust::device_ptr<Vec3> endOfUniqueVertices =
            thrust::unique(weldedVertexPointer, weldedVertexPointer + numVertices, VecEqual<Scalar,3>() );

    const uint numUniqueVertices = thrust::distance(weldedVertexPointer, endOfUniqueVertices);

    thrust::device_ptr<int> indexPointer(indices.data());

    thrust::device_ptr<const Vec3> originalVertexPointer(reinterpret_cast<const Vec3 *>(vertices.data()));

    thrust::lower_bound(weldedVertexPointer, endOfUniqueVertices,
                        originalVertexPointer, originalVertexPointer + numVertices,
                        indexPointer, VecLess<Scalar,3>() );

    return numUniqueVertices;
}

// compute colors of vertices
template <typename Scalar>
__global__ void computeColorsKernel(const Tensor<2, Scalar, DeviceResident> vertices,
                                    int* labels,
                                    unsigned char* class_colors,
                                    Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> min_val,
                                    Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> max_val,
                                    Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> offset,
                                    Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> scale,
                                    Tensor<2, unsigned char, DeviceResident> colors, int dimension, int num_classes) 
{
  const uint i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < vertices.dimensionSize(1))
  {
    // 3D point
    Scalar X = vertices(0, i) * scale(0) + offset(0);
    Scalar Y = vertices(1, i) * scale(1) + offset(1);
    Scalar Z = vertices(2, i) * scale(2) + offset(2);

    // voxel grid
    Scalar step_x = (max_val(0) - min_val(0)) / dimension;
    Scalar step_y = (max_val(1) - min_val(1)) / dimension;
    Scalar step_z = (max_val(2) - min_val(2)) / dimension;

    // grid location
    int x = std::round((X - min_val(0)) / step_x);
    int y = std::round((Y - min_val(1)) / step_y);
    int z = std::round((Z - min_val(2)) / step_z);

    if (x >= 0 && x < dimension && y >= 0 && y < dimension && z >= 0 && z < dimension)
    {
      int label = labels[x * dimension * dimension + y * dimension + z];
      colors(0, i) = class_colors[0 * num_classes + label];
      colors(1, i) = class_colors[1 * num_classes + label];
      colors(2, i) = class_colors[2 * num_classes + label];
    }
    else
    {
      colors(0, i) = 255;
      colors(1, i) = 255;
      colors(2, i) = 255;
    }
  }

}


template <typename Scalar, typename VoxelT>
void computeColors(const Tensor<2, Scalar, DeviceResident> & vertices, int* labels,
                   unsigned char* class_colors, const VoxelGrid<Scalar, VoxelT, DeviceResident> & voxelGrid,
                   Tensor<2, unsigned char, DeviceResident> & colors, int dimension, int num_classes)
{
  const uint numVertices = vertices.dimensionSize(1);

  const uint nThreads = 256;
  computeColorsKernel<<<intDivideAndCeil(numVertices, nThreads), nThreads>>>(vertices, labels, class_colors, voxelGrid.min(), voxelGrid.max(), 
    voxelGrid.gridToWorldOffset(), voxelGrid.gridToWorldScale(), colors, dimension, num_classes);
}

// extract labels from voxel grid

template <typename TransformerT,
          typename DepthCameraModelT,
          typename DepthT>
__global__ void computeLabelsKernel(const typename TransformerT::DeviceModule transformer,
                   const DepthCameraModelT depthCameraModel,
                   const DeviceTensor2<DepthT> depthMap,
                   const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident> voxelGrid,
                   DeviceTensor2<int> labels, 
                   DeviceTensor2<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > label_colors, 
                   const DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > class_colors, int width, int height) 
{
  typedef Eigen::Matrix<int,2,1,Eigen::DontAlign> Vec2i;

  const uint index = threadIdx.x + blockDim.x * blockIdx.x;

  const int x = index % width;
  const int y = index / width;
  if (x < width && y < height)
  {

    const Vec2i loc(x, y);
    DepthT depth = depthMap(loc);

    if (depth > 0)
    {
      // backprojection
      const Eigen::Matrix<float,2,1> point2d(x, y);
      const Eigen::Matrix<float,3,1> liveCoord = depthCameraModel.unproject(point2d, depth);
      const Eigen::Matrix<float,3,1> worldCoord = transformer.transformLiveToWorld(liveCoord);

      const Eigen::Matrix<float,3,1> gridCoord = voxelGrid.worldToGrid(worldCoord);
      int X = int(gridCoord(0));
      int Y = int(gridCoord(1));
      int Z = int(gridCoord(2));

      if (X >= 0 && X < voxelGrid.size(0) && Y >= 0 && Y < voxelGrid.size(1) && Z >= 0 && Z < voxelGrid.size(2))
      {
        CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> voxel = voxelGrid(X, Y, Z);
        Eigen::Matrix<float,10,1,Eigen::DontAlign> prob;
        prob = voxel.value<ProbabilityVoxel>();

        int label = -1;
        float max_prob = -1;
        for (int i = 0; i < 10; i++)
        {
          if (prob(i) > max_prob)
          {
            max_prob = prob(i);
            label = i;
          }
        }
        labels(loc) = label;
      }
      else
        labels(loc) = 0;
    }
    else
      labels(loc) = 0;
    // set color
    // label_colors(loc) = class_colors(labels(loc));
    label_colors(loc)(0) = 0;
    label_colors(loc)(1) = 0;
    label_colors(loc)(2) = 0;
  }
}

template <typename TransformerT,
          typename DepthCameraModelT,
          typename DepthT>
void computeLabels(const TransformerT & transformer,
                   const DepthCameraModelT & depthCameraModel,
                   const DeviceTensor2<DepthT> & depthMap,
                   const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident> & voxelGrid,
                   DeviceTensor2<int> & labels, 
                   DeviceTensor2<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & label_colors, 
                   const DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & class_colors)
{
  const uint width = depthMap.width();
  const uint height = depthMap.height();

  const uint output_size = width * height;
  const uint nThreads = 256;

  computeLabelsKernel<TransformerT,DepthCameraModelT,DepthT><<<intDivideAndCeil(output_size, nThreads), nThreads>>>
    (transformer.deviceModule(), depthCameraModel, depthMap, voxelGrid, labels, label_colors, class_colors, width, height);
}



// instances
template void extractSurface(ManagedTensor<2,float,DeviceResident> &,
                             const VoxelGrid<float,CompositeVoxel<float,TsdfVoxel>,DeviceResident> &,
                             const float);

template void extractSurface(ManagedTensor<2,float,DeviceResident> &,
                             const VoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel>,DeviceResident> &,
                             const float);

template void extractSurface(ManagedTensor<2,float,DeviceResident> &,
                             const VoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>,DeviceResident> &,
                             const float);

template uint weldVertices(const Tensor<2,float,DeviceResident> &,
                           Tensor<2,float,DeviceResident> &,
                           ManagedTensor<1,int,DeviceResident> &);

template void computeColors(const Tensor<2, float, DeviceResident> &, int*,
                   unsigned char*, const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel>, DeviceResident> &,
                   Tensor<2, unsigned char, DeviceResident> &, int, int);

template void computeColors(const Tensor<2, float, DeviceResident> &, int*,
                   unsigned char*, const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ColorVoxel>, DeviceResident> &,
                   Tensor<2, unsigned char, DeviceResident> &, int, int);

template void computeColors(const Tensor<2, float, DeviceResident> &, int*,
                   unsigned char*, const VoxelGrid<float, CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident> &,
                   Tensor<2, unsigned char, DeviceResident> &, int, int);

template void computeLabels(const RigidTransformer<float> &,
                        const Poly3CameraModel<float> &,
                        const DeviceTensor2<float> &,
                        const VoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel>, DeviceResident> &,
                        DeviceTensor2<int> &, DeviceTensor2<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > &, 
                        const DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > &);

} // namespace df
