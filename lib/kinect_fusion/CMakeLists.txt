project("DynamicFusion")
cmake_minimum_required(VERSION 2.8)

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release)
endif()

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# supply compile-time directory to code
set(compileDirectory \"${PROJECT_SOURCE_DIR}\")
add_definitions(-DCOMPILE_DIR=${compileDirectory})

option(BUILD_POLY3 "build for poly3 camera model" on)
if(BUILD_POLY3)
  add_definitions(-DBUILD_POLY3)
endif()

option(BUILD_LINEAR "build for linear camera model" on)
if (BUILD_LINEAR)
  add_definitions(-DBUILD_LINEAR)
endif()

option(BUILD_DOUBLE "build for double precision" off)
if (BUILD_DOUBLE)
  add_definitions(-DBUILD_DOUBLE)
endif()

find_package(CUDA REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(nanoflann nanoflann REQUIRED)

# TODO:
set(SUITESPARSE_INCLUDE_DIRS "/usr/include/suitesparse" CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIBRARIES "cholmod;metis")
#set(EIGEN_DIR "/home/yuxiang/tensorflow/local/lib/python2.7/site-packages/tensorflow/include")

# suitesparse libraries /usr/lib/x86_64-linux-gnu/libspqr.so;/usr/lib/x86_64-linux-gnu/libcholmod.so;/usr/lib/x86_64-linux-gnu/libccolamd.so;/usr/lib/x86_64-linux-gnu/libcamd.so;/usr/lib/x86_64-linux-gnu/libcolamd.so;/usr/lib/x86_64-linux-gnu/libamd.so;/usr/lib/liblapack.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so;/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so;/usr/lib/x86_64-linux-gnu/librt.so;/usr/lib/x86_64-linux-gnu/libmetis.so


include_directories(${PROJECT_SOURCE_DIR}/include
                    ${Pangolin_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${SUITESPARSE_INCLUDE_DIRS}
                    ${nanoflann_INCLUDE_DIRS})

link_directories(${Pangolin_LIBRARY_DIRS})

link_libraries(${Pangolin_LIBRARIES}
               ${SUITESPARSE_LIBRARIES} "-lutil")

file( GLOB DF_HEADERS
  "./include/df/*/*.h"
)

file( GLOB DF_SOURCES
  "./src/*/*.cpp"
)

file( GLOB DF_CUDA_SOURCES
  "./src/*/*.cu"
)

file( GLOB SHADER_SOURCES
  "./shaders/*.vert"
  "./shaders/*.frag"
)

#set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;-Xcompiler -fPIC;")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-gencode arch=compute_61,code=sm_61;--expt-extended-lambda;--verbose;")

message(STATUS "pangolin: ${Pangolin_INCLUDE_DIRS}")

cuda_add_library(
  kfusion
  SHARED
  kinect_fusion.cpp
  ${DF_HEADERS}
  ${DF_SOURCES}
  ${DF_CUDA_SOURCES}
  ${SHADER_SOURCES}
)

cuda_add_executable(
  kinectFusion
  kinect_fusion.cpp
  ${DF_HEADERS}
  ${DF_SOURCES}
  ${DF_CUDA_SOURCES}
  ${SHADER_SOURCES}
)
