#include <df/surface/color.h>

#include <df/util/cudaHelpers.h>

namespace df {

// template <typename Scalar, typename VoxelT>
__global__
void computeSurfaceColorsKernel(const DeviceTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > vertices,
                                DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > colors,
                                const DeviceTensor3<CompositeVoxel<float,TsdfVoxel,ColorVoxel> > voxelGrid) {

    typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vec3;

    const int vertexIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (vertexIndex < vertices.length()) {

        const Vec3 & vertex = vertices(vertexIndex);
        Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> & c = colors(vertexIndex);
        Vec3 color;

        if (voxelGrid.inBounds(vertex,0.f)) {

            if (vertex(0) != floor(vertex(0))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> >(),vertex(0),(int)vertex(1),(int)vertex(2));

            } else if (vertex(1) != floor(vertex(1))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> >(),(int)vertex(0),vertex(1),(int)vertex(2));

            } else if (vertex(2) != floor(vertex(2))) {

                color = voxelGrid.transformInterpolate(ColorValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> >(),(int)vertex(0),(int)vertex(1),vertex(2));

            } else {

                color = voxelGrid(vertex.cast<int>()).value<ColorVoxel>();

            }

        } else {

            color = Vec3(0, 0, 0);

        }

        c(0) = (unsigned char)(255 * color(0));
        c(1) = (unsigned char)(255 * color(1));
        c(2) = (unsigned char)(255 * color(2));

    }

}

__global__
void computeSurfaceColorsKernel(const DeviceTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > vertices,
                                DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > colors,
                                const DeviceTensor3<CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> > voxelGrid) {

    typedef Eigen::Matrix<float,3,1,Eigen::DontAlign> Vec3;
    typedef Eigen::Matrix<float,10,1,Eigen::DontAlign> Vec;

    const int vertexIndex = threadIdx.x + blockDim.x * blockIdx.x;

    if (vertexIndex < vertices.length()) {

        const Vec3 & vertex = vertices(vertexIndex);
        Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> & c = colors(vertexIndex);
        Vec prob;

        if (voxelGrid.inBounds(vertex,0.f)) {

            if (vertex(0) != floor(vertex(0))) {

                prob = voxelGrid.transformInterpolate(ProbabilityValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >(),vertex(0),(int)vertex(1),(int)vertex(2));

            } else if (vertex(1) != floor(vertex(1))) {

                prob = voxelGrid.transformInterpolate(ProbabilityValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >(),(int)vertex(0),vertex(1),(int)vertex(2));

            } else if (vertex(2) != floor(vertex(2))) {

                prob = voxelGrid.transformInterpolate(ProbabilityValueExtractor<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> >(),(int)vertex(0),(int)vertex(1),vertex(2));

            } else {

                prob = voxelGrid(vertex.cast<int>()).value<ProbabilityVoxel>();

            }

        } else {

            prob = Eigen::Matrix<float,10,1,Eigen::DontAlign>::Zero();

        }

        c(0) = (unsigned char)(255 * 1);
        c(1) = (unsigned char)(255 * 1);
        c(2) = (unsigned char)(255 * 1);

    }

}

template <typename Scalar, typename VoxelT>
void computeSurfaceColors(const DeviceTensor1<Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> > & vertices,
                          DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > & colors,
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
                                   DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > &,
                                   const DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ColorVoxel> > &);

template void computeSurfaceColors(const DeviceTensor1<Eigen::Matrix<float,3,1,Eigen::DontAlign> > &,
                                   DeviceTensor1<Eigen::Matrix<unsigned char,3,1,Eigen::DontAlign> > &,
                                   const DeviceVoxelGrid<float,CompositeVoxel<float,TsdfVoxel,ProbabilityVoxel> > &);


} // namespace df
