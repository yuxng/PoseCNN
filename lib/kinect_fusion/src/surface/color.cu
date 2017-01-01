#include <df/surface/color.h>

#include <df/util/cudaHelpers.h>

namespace df {

template <typename Scalar, typename VoxelT>
__global__
void computeSurfaceColorsKernel(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > vertices,
                                DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > colors,
                                const DeviceTensor3<VoxelT> voxelGrid) {

    typedef Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> Vec3;

    const int vertexIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (vertexIndex < vertices.length()) {

        const Vec3 & vertex = vertices(vertexIndex);
        Vec3 & color = colors(vertexIndex);

        if (voxelGrid.inBounds(vertex,0.f)) {

            if (vertex(0) != floor(vertex(0))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<Scalar,VoxelT>(),vertex(0),(int)vertex(1),(int)vertex(2));

            } else if (vertex(1) != floor(vertex(1))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<Scalar,VoxelT>(),(int)vertex(0),vertex(1),(int)vertex(2));

            } else if (vertex(2) != floor(vertex(2))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<Scalar,VoxelT>(),(int)vertex(0),(int)vertex(1),vertex(2));

            } else {

                color = voxelGrid(vertex.template cast<int>()).template value<ColorVoxel>();

            }

        } else {

            color = Vec3(0.5,0.5,0.5);

        }

    }

}

template <typename Scalar, typename VoxelT>
void computeSurfaceColors(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
                          DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & colors,
                          const DeviceVoxelGrid<Scalar,VoxelT> & voxelGrid) {

    const int numVertices = vertices.length();

    if (!numVertices) {
        return;
    }

    assert(colors.length() == numVertices);

    const dim3 block(1024);
    const dim3 grid(intDivideAndCeil((uint)numVertices,block.x));

    computeSurfaceColorsKernel<<<grid,block>>>(vertices,colors,voxelGrid.grid());

}


template void computeSurfaceColors(const DeviceTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > &,
                                   DeviceTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > &,
                                   const DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> > &);


} // namespace df
