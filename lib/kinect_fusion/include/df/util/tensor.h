#pragma once

#include <assert.h>

#include <iostream> // TODO

#include <Eigen/Core>

#include <cuda_runtime.h>

#include <df/util/tupleHelpers.h>
//#include <df/util/typeList.h>

namespace df {

enum Residency {
    HostResident,
    DeviceResident
};

namespace internal {

// -=-=-=- const qualification -=-=-=-
template <typename T, bool ConstQualified>
struct ConstQualifier;

template <typename T>
struct ConstQualifier<T,false> {
    typedef T type;
};

template <typename T>
struct ConstQualifier<T,true> {
    typedef const T type;
};

template <typename T>
struct ConstQualifier<T *,false> {
    typedef T * type;
};

template <typename T>
struct ConstQualifier<T *,true> {
    typedef const T * type;
};

// -=-=-=- copying -=-=-=-

template <Residency DestR, Residency SrcR>
struct CopyTypeTraits;

template <>
struct CopyTypeTraits<HostResident,DeviceResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyDeviceToHost;
};

template <>
struct CopyTypeTraits<DeviceResident,HostResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyHostToDevice;
};

template <>
struct CopyTypeTraits<DeviceResident,DeviceResident> {
    static constexpr cudaMemcpyKind copyType = cudaMemcpyDeviceToDevice;
};


template <typename T, Residency DestR, Residency SrcR>
struct Copier {

    inline static void copy(T * dst, const T * src, const std::size_t N) {
        cudaMemcpy(dst,src,N*sizeof(T),CopyTypeTraits<DestR,SrcR>::copyType);
    }

};

template <typename T>
struct Copier<T,HostResident,HostResident> {

    inline static void copy(T * dst, const T * src, const std::size_t N) {
        std::memcpy(dst,src,N*sizeof(T));
    }

};

// -=-=-=- size equivalence checking -=-=-=-
template <bool Check>
struct EquivalenceChecker {

    template <typename DimT, uint D>
    inline static void checkEquivalentSize(const Eigen::Matrix<DimT,D,1> & /*sizeA*/, const Eigen::Matrix<DimT,D,1> & /*sizeB*/) { }

    template <typename T>
    inline static void checkEquivalence(const T & /*A*/, const T & /*B*/) { }

};

template <>
struct EquivalenceChecker<true> {

    template <typename DimT, uint D>
    __attribute__((optimize("unroll-loops")))
    inline static void checkEquivalentSize(const Eigen::Matrix<DimT,D,1> & sizeA, const Eigen::Matrix<DimT,D,1> & sizeB) {
        for (int d=0; d<D; ++d) {
            if (sizeA(d) != sizeB(d)) {
                throw std::runtime_error("sizes in dimension " + std::to_string(d) + " do not match: " +
                                         std::to_string(sizeA(d)) + " vs " + std::to_string(sizeB(d)));
            }
        }
    }

    template <typename T>
    inline static void checkEquivalence(const T & A, const T & B) {
        if (A != B) {
            throw std::runtime_error("not equivalent");
        }
    }

};

// -=-=-=- automatic allocation -=-=-=-
template <typename T, Residency R>
struct AutomaticAllocator;

template <typename T>
struct AutomaticAllocator<T,HostResident> {

    inline static T * allocate(const std::size_t length) {
        T * vals = new T[length];
        return vals;
    }

    inline static void deallocate(T * vec) {
        delete [] vec;
    }

};

template <typename T>
struct AutomaticAllocator<T,DeviceResident> {

    inline static T * allocate(const std::size_t length) {
        T * vals;
        cudaMalloc(&vals,length*sizeof(T));
        return vals;
    }

    inline static void deallocate(T * vec) {
        cudaFree(vec);
    }

};

// -=-=-=- generic indexing -=-=-=-
template <typename T, int D>
struct IndexList {
    T head;
    IndexList<T,D-1> tail;

    __host__ __device__
    inline IndexList(const Eigen::Matrix<T,D,1> & indices)
        : head(indices(0)), tail(indices.template tail<D-1>()) { }

//    template <int D2>
//    __host__ __device__
//    inline IndexList(const Eigen::VectorBlock<const Eigen::Matrix<T,D2,1>,D> & indices))

    __host__ __device__
    inline T sum() const {
        return head + tail.sum();
    }

    __host__ __device__
    inline T product() const {
        return head * tail.product();
    }

};

//template <typename T>
//struct IndexList<T,1> {
//    T head;

//    __host__ __device__
//    inline IndexList(const Eigen::Matrix<T,1,1> & indices)
//        : head(indices(0)) { }

//    __host__ __device__
//    inline T sum() const {
//        return head;
//    }

//    __host__ __device__
//    inline T product() const {
//        return head;
//    }

//};

template <typename T>
struct IndexList<T,0> {

    __host__ __device__
    inline IndexList(const Eigen::Matrix<T,0,1> & indices) { }

    __host__ __device__
    inline T sum() const {
        return 0;
    }

    __host__ __device__
    inline T product() const {
        return 1;
    }

};

template <typename T>
inline __host__ __device__ IndexList<T,1> IndexList1(const T i0) {
    return { i0, IndexList<T,0>() };
}

template <typename T>
inline __host__ __device__ IndexList<T,2> IndexList2(const T i0, const T i1) {
    return { i0, IndexList1(i1) };
}

template <typename T>
inline __host__ __device__ IndexList<T,3> IndexList3(const T i0, const T i1, const T i2) {
    return { i0, IndexList2(i1, i2) };
}

template <typename T>
inline __host__ __device__ IndexList<T,4> IndexList4(const T i0, const T i1, const T i2, const T i3) {
    return { i0, IndexList3(i1, i2, i3) };
}

template <typename IdxT, typename DimT, int D>
inline __host__ __device__ std::size_t offsetXD(const IndexList<IdxT,D> dimIndices, const IndexList<DimT,D-1> dimSizes) {

    return dimIndices.head + dimSizes.head*offsetXD(dimIndices.tail,dimSizes.tail);

}

template <typename IdxT, typename DimT>
inline __host__ __device__ std::size_t offsetXD(const IndexList<IdxT,1> dimIndices, const IndexList<DimT,0> dimSizes) {

    return dimIndices.head;

}



//template <typename IdxT, typename DimT>
//inline __host__ __device__ std::size_t offsetXD(const IndexList<IdxT,2> dimIndices, const IndexList<DimT,1> dimSizes) {

//    return dimIndices.head + dimSizes.head*dimIndices.tail.head;

//}





// -=-=-=- interpolation -=-=-=-

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2;

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2<Scalar, float, IdxTs...> {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions,
//                                     float firstIndex, IdxTs ... remainingIndices) {

//        const uint i = firstIndex;
//        const float t = firstIndex - i;

//        return (1-t)*Interpolator2<Scalar, IdxTs...>::interpolate(data + i*dimensions.template head<Length-1>().prod(),
//                                                                  dimensions.template head<Length-1>(),
//                                                                  remainingIndices...)
//               + t * Interpolator2<Scalar, IdxTs...>::interpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                                                                  dimensions.template head<Length-1>(),
//                                                                  remainingIndices...);

//    }

//};

//template <typename Scalar, typename ... IdxTs>
//struct Interpolator2<Scalar, int, IdxTs...> {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions,
//                                     int firstIndex, IdxTs ... remainingIndices) {

//        return Interpolator2<Scalar, IdxTs...>::interpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                                                            dimensions.template head<Length-1>(),
//                                                            remainingIndices...);

//    }

//};

//template <typename Scalar>
//struct Interpolator2<Scalar> {

//    static constexpr uint Length = 0;

//    __host__ __device__
//    static inline Scalar interpolate(const Scalar * data,
//                                     const Eigen::Matrix<uint,Length,1> dimensions) {

//        return *data;

//    }

//};

//template <typename Scalar>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,0,1> dimensions) {

//    return *data;

//}

//template <typename Scalar, typename ... IdxTs>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,sizeof...(IdxTs)+1,1> dimensions,
//                          float firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    const uint i = firstIndex;
//    const float t = firstIndex - i;

//    return (1-t)*interpolate(data + i*dimensions.template head<Length-1>().prod(),
//                             dimensions.template head<Length-1>(),
//                             remainingIndices...)
//           + t * interpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                             dimensions.template head<Length-1>(),
//                             remainingIndices...);

//}

//template <typename Scalar, typename ... IdxTs>
//__host__ __device__
//inline Scalar interpolate(const Scalar * data,
//                          const Eigen::Matrix<uint,sizeof...(IdxTs) + 1,1> dimensions,
//                          int firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    return interpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                       dimensions.template head<Length-1>(),
//                       remainingIndices...);

//}


template <typename Scalar>
__host__ __device__
inline Scalar interpolate(const Scalar * data,
                          const IndexList<uint,0> /*dimensions*/,
                          const std::tuple<> /*remainingIndices*/) {

    return *data;

}

template <typename Scalar,typename ... IdxTs>
__host__ __device__
inline Scalar interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<float, IdxTs...> remainingIndices) {

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const float t = firstIndex - i;

    return (1-t)*interpolate(data + i*dimensions.tail.product(),
                             dimensions.tail,
                             tail(remainingIndices))
           + t * interpolate(data + (i+1)*dimensions.tail.product(),
                             dimensions.tail,
                             tail(remainingIndices));

}

template <typename Scalar, typename ... IdxTs>
__host__ __device__
inline Scalar interpolate(const Scalar * data,
                          const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                          const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return interpolate(data + firstIndex*dimensions.tail.product(),
                       dimensions.tail,
                       tail(remainingIndices));

}


// TODO: can this be subsumed into the original interpolate call by just having Transformer
// be the first type in the variadic parameter pack??
// the only tricky part would be deducing the return type through the recursive calls
//template <typename Scalar, typename Transformer>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,0,1> dimensions,
//                                                             Transformer transformer) {

//    return transformer(*data);

//}

//template <typename Scalar, typename Transformer, typename ... IdxTs>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,sizeof...(IdxTs)+1,1> dimensions,
//                                                             Transformer transformer,
//                                                             float firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    const uint i = firstIndex;
//    const float t = firstIndex - i;

//    return (1-t)*transformInterpolate(data + i*dimensions.template head<Length-1>().prod(),
//                                      dimensions.template head<Length-1>(),
//                                      transformer,
//                                      remainingIndices...)
//           + t * transformInterpolate(data + (i+1)*dimensions.template head<Length-1>().prod(),
//                                      dimensions.template head<Length-1>(),
//                                      transformer,
//                                      remainingIndices...);

//}

//template <typename Scalar, typename Transformer, typename ... IdxTs>
//__host__ __device__
//inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
//                                                             const Eigen::Matrix<uint,sizeof...(IdxTs) + 1,1> dimensions,
//                                                             Transformer transformer,
//                                                             int firstIndex, IdxTs ... remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

//    return transformInterpolate(data + firstIndex*dimensions.template head<Length-1>().prod(),
//                                dimensions.template head<Length-1>(),
//                                transformer,
//                                remainingIndices...);

//}

template <typename Scalar, typename Transformer>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,0> /*dimensions*/,
                                                             Transformer transformer,
                                                             const std::tuple<> /*remainingIndices*/) {

    return transformer(*data);

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                             Transformer transformer,
                                                             const std::tuple<float, IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const typename Transformer::ScalarType t = firstIndex - i;

    return (1-t)*transformInterpolate(data + i*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      tail(remainingIndices))
           + t * transformInterpolate(data + (i+1)*dimensions.tail.product(),
                                      dimensions.tail,
                                      transformer,
                                      tail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolate(const Scalar * data,
                                                             const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                             Transformer transformer,
                                                             const std::tuple<int, IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const int firstIndex = std::get<0>(remainingIndices);

    return transformInterpolate(data + firstIndex*dimensions.tail.product(),
                                dimensions.tail,
                                transformer,
                                tail(remainingIndices));

}


template <typename Scalar, typename Transformer, typename ValidityCheck>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,0> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<>) {

    if (check(*data)) {

        totalWeight += thisWeight;
        return thisWeight * transformer(*data);

    } else {

        return 0;

    }

}


template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<float,IdxTs...> remainingIndices) {

//    static constexpr uint Length = sizeof...(IdxTs) + 1;

    const float firstIndex = std::get<0>(remainingIndices);
    const uint i = firstIndex;
    const typename Transformer::ScalarType t = firstIndex - i;

    return transformInterpolateValidOnly(data + i*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * (1-t),
                                         transformer,
                                         check,
                                         tail(remainingIndices)) +
           transformInterpolateValidOnly(data + (i+1)*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight * t,
                                         transformer,
                                         check,
                                         tail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
inline typename Transformer::ReturnType transformInterpolateValidOnly(const Scalar * data,
                                                                      const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                                                      typename Transformer::ScalarType & totalWeight,
                                                                      const typename Transformer::ScalarType thisWeight,
                                                                      Transformer transformer,
                                                                      ValidityCheck check,
                                                                      const std::tuple<int, IdxTs...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return transformInterpolateValidOnly(data + firstIndex*dimensions.tail.product(),
                                         dimensions.tail,
                                         totalWeight,
                                         thisWeight,
                                         transformer,
                                         check,
                                         tail(remainingIndices));

}


template <typename Scalar, typename ValidityCheck>
__host__ __device__
inline bool validForInterpolation(const Scalar * data,
                                   const IndexList<uint,0> dimensions,
                                   ValidityCheck check,
                                   const std::tuple<> ) {

    return check(*data);

}

template <typename Scalar, typename ValidityCheck, typename ... IdxTs>
__host__ __device__
inline bool validForInterpolation(const Scalar * data,
                                  const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                  ValidityCheck check,
                                  const std::tuple<float,IdxTs ...> remainingIndices) {

    const int i = std::get<0>(remainingIndices);

    return validForInterpolation(data + i*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 tail(remainingIndices)) &&
           validForInterpolation(data + (i+1)*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 tail(remainingIndices));

}

template <typename Scalar, typename Transformer, typename ... IdxTs>
__host__ __device__
inline bool validForInterpolation(const Scalar * data,
                                 const IndexList<uint,sizeof...(IdxTs)+1> dimensions,
                                 Transformer check,
                                 const std::tuple<int, IdxTs ...> remainingIndices) {

    const int firstIndex = std::get<0>(remainingIndices);

    return validForInterpolation(data + firstIndex*dimensions.tail.product(),
                                 dimensions.tail,
                                 check,
                                 tail(remainingIndices));

}


//template <typename Scalar,
//          typename Head,
//          typename Tail>
//struct Interpolator {

//};

//template <typename Scalar,
//          typename Tail>
//struct Interpolator<Scalar,float,Tail> {

//    typedef TypeList<float,Tail> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __host__ __device__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,IndexTypeList::Length,1> dimensions) {

//        const uint i = indices.head;
//        const float t = indices.head - i;

//        return (1-t)*Interpolator<Scalar,typename Tail::Head,typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + i*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>())
//               + t * Interpolator<Scalar,typename Tail::Head,typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + (i+1)*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>());

//    }

//};

//template <typename Scalar,
//          typename Tail>
//struct Interpolator<Scalar,int,Tail> {

//    typedef TypeList<int,Tail> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __host__ __device__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> dimensions) {

//        return Interpolator<Scalar,typename Tail::Head, typename Tail::Tail>
//                ::interpolate(indices.tail,
//                              data + indices.head*dimensions.template head<Length-1>().prod(),
//                              dimensions.template head<Length-1>());

//    }

//};

//template <typename Scalar>
//struct Interpolator<Scalar,float,NullType> {

//    typedef TypeList<float,NullType> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __host__ __device__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> /*dimensions*/) {

//        const uint i = indices.head;
//        const float t = indices.head - i;

//        return (1-t) * data[i] + t * data[i + 1];

//    }

//};

//template <typename Scalar>
//struct Interpolator<Scalar,int,NullType> {

//    typedef TypeList<int,NullType> IndexTypeList;
//    static constexpr uint Length = IndexTypeList::Length;

//    __host__ __device__ static
//    inline Scalar interpolate(const CompositedTypeListInstantiation<IndexTypeList> indices,
//                              const Scalar * data,
//                              const Eigen::Matrix<uint,Length,1> /*dimensions*/) {

//        return data[indices.head];

//    }

//};


} // namespace internal

template <uint D, typename T, Residency R = HostResident, bool Const = false>
class Tensor {
public:

    typedef unsigned int DimT;
    typedef unsigned int IdxT;

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    __host__ __device__ Tensor(const DimT length) : dimensions_(Eigen::Matrix<DimT,D,1>(length)), data_(nullptr) { }

    __host__ __device__ Tensor(const Eigen::Matrix<DimT,D,1> & dimensions) : dimensions_(dimensions), data_(nullptr) { }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    __host__ __device__ Tensor(const DimT length, typename internal::ConstQualifier<T *,Const>::type data) :
        dimensions_(Eigen::Matrix<DimT,D,1>(length)), data_(data) { }

    // construct with values, not valid for managed tensors
    __host__ __device__ Tensor(const Eigen::Matrix<DimT,D,1> & dimensions,
                               typename internal::ConstQualifier<T *,Const>::type data) : dimensions_(dimensions), data_(data) { }

    // copy constructor and assignment operator, not valid for managed or const tensors
    template <bool _Const>
    __host__ __device__  Tensor(Tensor<D,T,R,_Const> & other)
        : dimensions_(other.dimensions()), data_(other.data()) {
        static_assert(Const || !_Const,
                      "Cannot copy-construct a non-const Tensor from a Const tensor");
    }

    template <bool _Const>
    __host__ __device__ inline Tensor<D,T,R,Const> & operator=(const Tensor<D,T,R,_Const> & other) {
        static_assert(Const || !_Const,
                      "Cannot assign a non-const Tensor from a Const tensor");
        dimensions_ = other.dimensions();
        data_ = other.data();
        return *this;
    }

    __host__ __device__ ~Tensor() { }

    // conversion to const tensor
    template <bool _Const = Const, typename std::enable_if<!_Const,int>::type = 0>
    inline operator Tensor<D,T,R,true>() const {
        return Tensor<D,T,R,true>( dimensions(), data() );
    }

    template <typename U = T,
              typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __host__ __device__ T * data() { return data_; }

    inline __host__ __device__ const T * data() const { return data_; }

    // -=-=-=-=-=-=- sizing functions -=-=-=-=-=-=-
    inline __host__ __device__ DimT dimensionSize(const IdxT dim) const {
        return dimensions_(dim);
    }

    inline __host__ __device__ const Eigen::Matrix<DimT,D,1> & dimensions() const {
        return dimensions_;
    }

    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ DimT length() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ DimT width() const {
        return dimensions_(0);
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ DimT height() const {
        return dimensions_(1);
    }

    inline __host__ __device__ std::size_t count() const {
//        return internal::count<DimT,D>(dimensions_);
        return dimensions_.prod();
    }

    inline __host__ __device__ std::size_t sizeBytes() const {
        return count() * sizeof(T);
    }

    // -=-=-=-=-=-=- indexing functions -=-=-=-=-=-=-
    template <int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0) const {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 1 && !Const, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0) {
        return data_[d0];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1) const {
        return data_[internal::offsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1) {
        return data_[internal::offsetXD<IdxT,DimT,2>(internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(d0,d1)),
                internal::IndexList<IdxT,1>(Eigen::Matrix<uint,1,1>(dimensions_[0])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1));
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1, const IdxT d2) const {
        return data_[internal::offsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1, const IdxT d2) {
        return data_[internal::offsetXD<IdxT,DimT,3>(internal::IndexList<IdxT,3>(Eigen::Matrix<uint,3,1>(d0,d1,d2)),
                internal::IndexList<IdxT,2>(Eigen::Matrix<uint,2,1>(dimensions_[0],dimensions_[1])))];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1),indices(2));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1),indices(2));
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ const T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) const {
        return data_[internal::offsetXD<IdxT,DimT,4>({d0,d1,d2,d3},{dimensions_[0],dimensions_[1],dimensions_[2]})];
    }

    template <int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ T & operator()(const IdxT d0, const IdxT d1, const IdxT d2, const IdxT d3) {
        return data_[internal::offsetXD<IdxT,DimT,4>({d0,d1,d2,d3},{dimensions_[0],dimensions_[1],dimensions_[2]})];
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ T & operator()(const Eigen::MatrixBase<Derived> & indices) {
        return operator()(indices(0),indices(1),indices(2),indices(3));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_integral<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ const T & operator()(const Eigen::MatrixBase<Derived> & indices) const {
        return operator()(indices(0),indices(1),indices(2),indices(3));
    }

    // -=-=-=-=-=-=- interpolation functions -=-=-=-=-=-=-
//    template <typename IdxT1,
//              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0) const {
//        return internal::interpolate(data_, dimensions_, v0);
//    }

//    template <typename IdxT1, typename IdxT2,
//              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1) const {
//        return internal::interpolate(data_, dimensions_, v1, v0);
//    }

//    template <typename IdxT1, typename IdxT2, typename IdxT3,
//              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) const {
//        return internal::interpolate(data_, dimensions_, v2, v1, v0);
//    }

//    template <typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
//              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
//    inline __host__ __device__ T interpolate(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) const {
//        return internal::interpolate(data_, dimensions_, v3, v2, v1, v0);
//    }

    template <typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ T interpolate(const IdxTs ... vs) const {
        return internal::interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ T interpolate(const Eigen::MatrixBase<Derived> & v) const {
        return internal::interpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()),
                                     vectorToTuple(v.reverse()));
    }

    template <typename Transformer, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ typename Transformer::ReturnType transformInterpolate(Transformer transformer, const IdxTs ... vs) const {
        return internal::transformInterpolate(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), transformer,
                                              internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    template <typename Transformer, typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ typename Transformer::ReturnType transformInterpolateValidOnly(Transformer transformer, ValidityCheck check, IdxTs ... vs) const {
        typename Transformer::ScalarType totalWeight(0);
        const typename Transformer::ScalarType totalValue = internal::transformInterpolateValidOnly(data_,internal::IndexList<DimT,D>(dimensions_.reverse()),
                                                                                                    totalWeight, typename Transformer::ScalarType(1), transformer, check,
                                                                                                    internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));

        if (totalWeight) {
            return totalValue / totalWeight;
        }

        return 0;

    }

    template <typename ValidityCheck, typename ... IdxTs,
              typename std::enable_if<sizeof...(IdxTs) == D, int>::type = 0>
    inline __host__ __device__ bool validForInterpolation(ValidityCheck check, const IdxTs ... vs) {
        return internal::validForInterpolation(data_, internal::IndexList<DimT,D>(dimensions_.reverse()), check,
                                               internal::TupleReverser<std::tuple<IdxTs...> >::reverse(std::tuple<IdxTs...>(vs...)));
    }

    // -=-=-=-=-=-=- bounds-checking functions -=-=-=-=-=-=-
    template <typename PosT, int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ bool inBounds(const PosT d0, const PosT border) const {
        return (d0 >= border) && (d0 <= dimensionSize(0) - 1 - border);
    }

    template <typename PosT, int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ bool inBounds(const PosT d0, const PosT d1, const PosT border) const {
        return (d0 >= border) && (d0 <= dimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= dimensionSize(1) - 1 - border);
    }

    template <typename PosT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool inBounds(const Eigen::MatrixBase<Derived> & point, const PosT border) const {
        return inBounds(point(0),point(1),border);
    }

    template <typename PosT, int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ bool inBounds(const PosT d0, const PosT d1, const PosT d2, const PosT border) const {
        return (d0 >= border) && (d0 <= dimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= dimensionSize(1) - 1 - border) &&
               (d2 >= border) && (d2 <= dimensionSize(2) - 1 - border);
    }

    template <typename PosT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool inBounds(const Eigen::MatrixBase<Derived> & point, const PosT border) const {
        return inBounds(point(0),point(1),point(2),border);
    }

    template <typename PosT, int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ bool inBounds(const PosT d0, const PosT d1, const PosT d2, const PosT d3, const PosT border) const {
        return (d0 >= border) && (d0 <= dimensionSize(0) - 1 - border) &&
               (d1 >= border) && (d1 <= dimensionSize(1) - 1 - border) &&
               (d2 >= border) && (d2 <= dimensionSize(2) - 1 - border) &&
               (d3 >= border) && (d3 <= dimensionSize(3) - 1 - border);
    }

    template <typename PosT, typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                      std::is_arithmetic<typename Eigen::internal::traits<Derived>::Scalar>::value, int>::type = 0>
    inline __host__ __device__ bool inBounds(const Eigen::MatrixBase<Derived> & point, const PosT border) const {
        return inBounds(point(0),point(1),point(2),point(3),border);
    }

    // -=-=-=-=-=-=- gradient functions -=-=-=-=-=-=-
    template <typename IdxT1,
              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<T,D,1> backwardGradient(const IdxT1 v0) const {
        const T center = interpolate(v0);
        return Eigen::Matrix<T,D,1>(center - interpolate(v0-1));
    }

    template <typename IdxT1, typename IdxT2,
              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<T,D,1> backwardGradient(const IdxT1 v0, const IdxT2 v1) const {
        const T center = interpolate(v0,v1);
        return Eigen::Matrix<T,D,1>(center - interpolate(v0-1,v1),
                                    center - interpolate(v0,v1-1));
    }

    template <typename IdxT1, typename IdxT2, typename IdxT3,
              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<T,D,1> backwardGradient(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) const {
        const T center = interpolate(v0,v1,v2);
        return Eigen::Matrix<T,D,1>(center - interpolate(v0-1,v1,v2),
                                    center - interpolate(v0,v1-1,v2),
                                    center - interpolate(v0,v1,v2-1));
    }

    template <typename Derived,
              typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == D &&
                                      Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<T,D,1> backwardGradient(const Eigen::MatrixBase<Derived> & v) const {
        return backwardGradient(v(0),v(1),v(2));
    }

    template <typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<T,D,1> backwardGradient(const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) const {
        const T center = interpolate(v0,v1,v2,v3);
        return Eigen::Matrix<T,D,1>(center - interpolate(v0-1,v1,v2,v3),
                                    center - interpolate(v0,v1-1,v2,v3),
                                    center - interpolate(v0,v1,v2-1,v3),
                                    center - interpolate(v0,v1,v2,v3-1));
    }



    template <typename Transformer, typename IdxT1,
              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradient(Transformer transformer, const IdxT1 v0) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolate(transformer,v0);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolate(transformer,v0-1));
    }

    template <typename Transformer, typename IdxT1, typename IdxT2,
              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradient(Transformer transformer, const IdxT1 v0, const IdxT2 v1) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolate(transformer,v0,v1);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolate(transformer,v0-1,v1),
                                              center - transformInterpolate(transformer,v0,v1-1));
    }

    template <typename Transformer, typename IdxT1, typename IdxT2, typename IdxT3,
              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradient(Transformer transformer, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolate(transformer,v0,v1,v2);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolate(transformer,v0-1,v1,v2),
                                              center - transformInterpolate(transformer,v0,v1-1,v2),
                                              center - transformInterpolate(transformer,v0,v1,v2-1));
    }

    template <typename Transformer, typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradient(Transformer transformer, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolate(transformer,v0,v1,v2,v3);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolate(transformer,v0-1,v1,v2,v3),
                                              center - transformInterpolate(transformer,v0,v1-1,v2,v3),
                                              center - transformInterpolate(transformer,v0,v1,v2-1,v3),
                                              center - transformInterpolate(transformer,v0,v1,v2,v3-1));
    }


    template <typename Transformer, typename ValidityCheck, typename IdxT1,
              int D2 = D, typename std::enable_if<D2 == 1, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolateValidOnly(transformer,check,v0);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1));
    }

    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2,
              int D2 = D, typename std::enable_if<D2 == 2, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1));
    }

    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2, typename IdxT3,
              int D2 = D, typename std::enable_if<D2 == 3, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1,v2);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1,v2),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1,v2),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2-1));
    }

    template <typename Transformer, typename ValidityCheck, typename IdxT1, typename IdxT2, typename IdxT3, typename IdxT4,
              int D2 = D, typename std::enable_if<D2 == 4, int>::type = 0>
    inline __host__ __device__ Eigen::Matrix<typename Transformer::ReturnType,D,1> transformBackwardGradientValidOnly(Transformer transformer, ValidityCheck check, const IdxT1 v0, const IdxT2 v1, const IdxT3 v2, const IdxT4 v3) {
        typedef typename Transformer::ReturnType Transformed;
        const Transformed center = transformInterpolateValidOnly(transformer,check,v0,v1,v2,v3);
        return Eigen::Matrix<Transformed,D,1>(center - transformInterpolateValidOnly(transformer,check,v0-1,v1,v2,v3),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1-1,v2,v3),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2-1,v3),
                                              center - transformInterpolateValidOnly(transformer,check,v0,v1,v2,v3-1));
    }


    // -=-=-=-=-=-=- pointer manipulation functions -=-=-=-=-=-=-
    template <typename U = T,
              typename std::enable_if<!Const && sizeof(U), int>::type = 0>
    inline __host__ __device__ void setDataPointer(T * data) { data_ = data; }

    // -=-=-=-=-=-=- copying functions -=-=-=-=-=-=-
    template <Residency R2, bool Const2, bool Check=false>
    inline void copyFrom(const Tensor<D,T,R2,Const2> & other) {
        static_assert(!Const,"you cannot copy to a const tensor");
        internal::EquivalenceChecker<Check>::template checkEquivalentSize<DimT,D>(dimensions(),other.dimensions());
        internal::Copier<T,R,R2>::copy(data_,other.data(),count());
    }

protected:

    Eigen::Matrix<DimT,D,1> dimensions_;
    typename internal::ConstQualifier<T *,Const>::type data_;

};


template <uint D, typename T, Residency R = HostResident>
class ManagedTensor : public Tensor<D,T,R,false> {
public:

    typedef typename Tensor<D,T,R,false>::DimT DimT;

    ManagedTensor() :
        Tensor<D,T,R,false>::Tensor(Eigen::Matrix<uint,D,1>::Zero(), nullptr) { }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    ManagedTensor(const DimT length) :
        Tensor<D,T,R,false>::Tensor(length, internal::AutomaticAllocator<T,R>::allocate(length)) { }

    ManagedTensor(const Eigen::Matrix<DimT,D,1> & dimensions) :
        Tensor<D,T,R,false>::Tensor(dimensions,
                                    internal::AutomaticAllocator<T,R>::allocate(dimensions.prod())) { }

    ~ManagedTensor() {
        internal::AutomaticAllocator<T,R>::deallocate(this->data_);
    }

    template <int D2 = D, typename std::enable_if<D2 == 1,int>::type = 0>
    inline void resize(const DimT length) {
        resize(Eigen::Matrix<DimT,D,1>(length));
    }

    void resize(const Eigen::Matrix<DimT,D,1> & dimensions) {
        internal::AutomaticAllocator<T,R>::deallocate(this->data_);
        this->data_ = internal::AutomaticAllocator<T,R>::allocate(dimensions.prod());
        this->dimensions_ = dimensions;
    }

private:

    ManagedTensor(const ManagedTensor &) = delete;
    ManagedTensor & operator=(const ManagedTensor &) = delete;

};

//namespace internal {

//typedef unsigned int DimT;

//template <bool Packed>
//struct FirstDimensionStride;

//template <>
//struct FirstDimensionStride<true> {
//    inline DimT stride() const { return 1; }
//};

//template <>
//struct FirstDimensionStride<false> {
//    inline DimT stride() const { return stride_; }
//    DimT stride_;
//};


//template <bool SourcePacked, unsigned int FirstDimension>
//struct SliceReturnValPacked {
////    using Determinant = PackingDeterminant<SourcePacked>::Determinant;
////    static constexpr bool Packed = Determinant<FirstDimension>::Packed;
//    static constexpr bool Packed = false;
//};

//template <>
//struct SliceReturnValPacked<true,0> {
//    static constexpr bool Packed = true;
//};


//} // namespace internal

//template <uint D, typename T, Residency R = HostResident, bool Const = false, bool Packed = true>
//class Tensor {
//public:

//    typedef internal::DimT DimT;
//    typedef unsigned int IndT;

//    inline __host__ __device__ DimT dimensionSize(const IndT dim) const {
//        assert(dim < D);
//        return dimensions_[dim];
//    }

//    template <unsigned int FirstDimension, unsigned int ... Rest>
//    inline __host__ __device__ Tensor<D,T,R,Const,internal::SliceReturnValPacked<Packed,FirstDimension>::Packed> slice() {

//    }

//protected:

//    std::array<DimT,D> dimensions_;
//    internal::FirstDimensionStride<Packed> firstDimensionStride_;
//    std::array<DimT,D-1> otherDimensionStrides_;
//    typename ConstQualifier<T *,Const>::type values_;

//};

// -=-=-=-=- full tensor typedefs -=-=-=-=-
#define TENSOR_TYPEDEFS_(i, type, appendix)                                       \
    typedef Tensor<i,type,HostResident> Tensor##i##appendix;                      \
    typedef Tensor<i,type,DeviceResident> DeviceTensor##i##appendix;              \
    typedef Tensor<i,type,HostResident,true> ConstTensor##i##appendix;            \
    typedef Tensor<i,type,DeviceResident,true> ConstDeviceTensor##i##appendix;    \
    typedef ManagedTensor<i,type,HostResident> ManagedTensor##i##appendix;        \
    typedef ManagedTensor<i,type,DeviceResident> ManagedDeviceTensor##i##appendix

#define TENSOR_TYPEDEFS(type, appendix)  \
    TENSOR_TYPEDEFS_(1, type, appendix); \
    TENSOR_TYPEDEFS_(2, type, appendix); \
    TENSOR_TYPEDEFS_(3, type, appendix); \
    TENSOR_TYPEDEFS_(4, type, appendix)

TENSOR_TYPEDEFS(float,f);
TENSOR_TYPEDEFS(int,i);
TENSOR_TYPEDEFS(uint,ui);

template <int D, typename Scalar>
using DeviceTensor = Tensor<D,Scalar,DeviceResident>;

template <int D, typename Scalar>
using ConstTensor = Tensor<D,Scalar,HostResident, true>;

template <int D, typename Scalar>
using ConstDeviceTensor = Tensor<D,Scalar,DeviceResident, true>;

#define TENSOR_PARTIAL_TYPEDEF_(i,residency)                                         \
    template <typename Scalar>                                                       \
    using residency##Tensor##i = Tensor<i,Scalar,residency##Resident>;               \
    template <typename Scalar>                                                       \
    using Const##residency##Tensor##i = Tensor<i,Scalar,residency##Resident,true>;   \
    template <typename Scalar>                                                       \
    using Managed##residency##Tensor##i = ManagedTensor<i,Scalar,residency##Resident>

#define TENSOR_PARTIAL_TYPEDEF(i)                  \
    TENSOR_PARTIAL_TYPEDEF_(i,Device);             \
    TENSOR_PARTIAL_TYPEDEF_(i,Host)

//template <typename Scalar>
//using DeviceTensor2 = Tensor<2,Scalar,DeviceResident>;

TENSOR_PARTIAL_TYPEDEF(1);
TENSOR_PARTIAL_TYPEDEF(2);
TENSOR_PARTIAL_TYPEDEF(3);
TENSOR_PARTIAL_TYPEDEF(4);




} // namespace df
