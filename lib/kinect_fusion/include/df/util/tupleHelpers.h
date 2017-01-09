#pragma once

#include <tuple>

#include <cuda_runtime.h>

#include <Eigen/Core>

namespace df {

namespace internal {

template <size_t... N>
struct IntegerListConstructor {
    template <size_t M>
    struct PushBack {
        typedef IntegerListConstructor<N...,M> type;
    };
};

template <size_t Max>
struct IntegerList {
    typedef typename IntegerList<Max-1>::type::template PushBack<Max>::type type;
};

template <>
struct IntegerList<0> {
    typedef IntegerListConstructor<> type;
};

template <size_t...Indices, typename Tuple>
__host__ __device__ inline auto tupleSubset(const Tuple & tuple, IntegerListConstructor<Indices...>)
    -> decltype(std::make_tuple(std::get<Indices>(tuple)...)) {
    return std::make_tuple(std::get<Indices>(tuple)...);
}

template <typename Head, typename ... Tail>
__host__ __device__
inline std::tuple<Tail...> tail(const std::tuple<Head,Tail...> & t) {

    return tupleSubset(t, typename IntegerList<sizeof...(Tail)>::type());

}


template <typename TupleT>
struct TupleReverser;

template <typename T>
struct TupleReverser<std::tuple<T> > {
    using Type = std::tuple<T>;

    static __host__ __device__ inline Type reverse(const std::tuple<T> & t) {
        return t;
    }

};

template <typename Head, typename ... Tail>
struct TupleReverser<std::tuple<Head,Tail...> > {

    using HeadTuple = std::tuple<Head>;
    using TailTuple = typename TupleReverser<std::tuple<Tail...> >::Type;

    using Type = decltype(std::tuple_cat(std::declval<TailTuple>(), std::declval<HeadTuple>()));

    static __host__ __device__ inline Type reverse(const std::tuple<Head,Tail...> & t) {

        return std::tuple_cat(TupleReverser<std::tuple<Tail...> >::reverse(tail(t)), std::tuple<Head>(std::get<0>(t)));

    }

};

template <typename QueryType, std::size_t CheckIndex, typename ... TupleTypes>
struct TupleIndexHunter;

template <typename QueryType, std::size_t CheckIndex, typename FirstType, typename ... TupleTypes>
struct TupleIndexHunter<QueryType,CheckIndex,FirstType,TupleTypes...> {

    static constexpr auto Index = TupleIndexHunter<QueryType,CheckIndex+1,TupleTypes...>::Index;

};

template <typename QueryType, std::size_t CheckIndex, typename ... TupleTypes>
struct TupleIndexHunter<QueryType,CheckIndex,QueryType,TupleTypes...> {

    static constexpr auto Index = CheckIndex;

};

template <std::size_t Index, typename ... TupleTypes>
struct DeviceExecutableTupleCopy {

    static inline __host__ __device__ void copy(std::tuple<TupleTypes...> & destination,
                                                const std::tuple<TupleTypes...> & source) {
        std::get<Index>(destination) = std::get<Index>(source);
        DeviceExecutableTupleCopy<Index-1,TupleTypes...>::copy(destination,source);
    }

};

template <typename ... TupleTypes>
struct DeviceExecutableTupleCopy<-1,TupleTypes...> {

    static inline __host__ __device__ void copy(std::tuple<TupleTypes...> & /*destination*/,
                                                const std::tuple<TupleTypes...> & /*source*/) {
//        std::get<0>(destination) = std::get<0>(source);
    }

};


template <typename Scalar, int D>
struct TupledType {

    using HeadTuple = std::tuple<Scalar>;
    using TailTuple = typename TupledType<Scalar,D-1>::Type;

    using Type = decltype(std::tuple_cat(std::declval<HeadTuple>(), std::declval<TailTuple>()));

};

template <typename Scalar>
struct TupledType<Scalar, 1> {

    using Type = std::tuple<Scalar>;

};


} // namespace internal

template <typename QueryType, typename ... TupleTypes>
__host__ __device__ inline
QueryType & getByType(std::tuple<TupleTypes...> & tuple) {

    return std::get<internal::TupleIndexHunter<QueryType,0,TupleTypes ...>::Index>(tuple);

}

template <typename QueryType, typename ... TupleTypes>
__host__ __device__ inline
const QueryType & getByType(const std::tuple<TupleTypes...> & tuple) {

    return std::get<internal::TupleIndexHunter<QueryType,0,TupleTypes ...>::Index>(tuple);

}

template <typename ... TupleTypes>
__host__ __device__ inline
void copy(std::tuple<TupleTypes...> & destination, const std::tuple<TupleTypes...> & source) {
    internal::DeviceExecutableTupleCopy<sizeof...(TupleTypes)-1,TupleTypes...>::copy(destination,source);
}


template <typename Derived,
          typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 1 &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
__host__ __device__ inline
typename internal::TupledType<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime>::Type
vectorToTuple(const Eigen::MatrixBase<Derived> & vector) {

    return std::tuple<typename Eigen::internal::traits<Derived>::Scalar>(vector(0));

}

template <typename Derived,
          typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime >= 2 &&
                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
__host__ __device__ inline
typename internal::TupledType<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime>::Type
vectorToTuple(const Eigen::MatrixBase<Derived> & vector) {

    return std::tuple_cat(std::tuple<typename Eigen::internal::traits<Derived>::Scalar>(vector(0)),
                          vectorToTuple(vector.template tail<Eigen::internal::traits<Derived>::RowsAtCompileTime-1>()));

}

} // namespace df
