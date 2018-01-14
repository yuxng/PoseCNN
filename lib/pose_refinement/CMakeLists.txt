project("refiner")
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include(${CMAKE_SOURCE_DIR}/cmake/FindNLopt.cmake)

find_package(CUDA REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
pkg_check_modules(OpenEXR REQUIRED OpenEXR)

set(xFusion_ROOT "/home/yuxiang/Projects/Deep_Pose/lib/kinect_fusion")
set(xFusion_INCLUDE_DIRS ${xFusion_ROOT}/include)
set(xFusion_LIBRARY_DIRS ${xFusion_ROOT}/build)
set(xFusion_LIBRARIES kfusion)

# TODO:
set(SUITESPARSE_INCLUDE_DIRS "/usr/include/suitesparse" CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIBRARIES "cholmod;metis")

message(STATUS "xFusion root: ${xFusion_ROOT}")
message(STATUS "pangolin: ${Pangolin_INCLUDE_DIRS}")

include_directories(${Pangolin_INCLUDE_DIRS}
                    ${xFusion_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${OpenEXR_INCLUDE_DIRS}
                    ${SUITESPARSE_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

link_directories(${Pangolin_LIBRARY_DIRS}
                 ${xFusion_LIBRARY_DIRS}
                 ${OpenEXR_LIBRARY_DIRS})

link_libraries(${Pangolin_LIBRARIES}
               ${xFusion_LIBRARIES}
               ${DEPTHSENSE_LIBRARIES}
               ${SUITESPARSE_LIBRARIES}
               ${OpenEXR_LIBRARIES}
               ${OpenCV_LIBS}
               ${NLOPT_LIBRARIES}
               assimp
               util)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;")

cuda_add_library(
  refiner
  SHARED
  refinement.cpp
)

cuda_add_executable(refinement
                    refinement.cpp)
