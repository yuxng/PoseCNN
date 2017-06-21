project("ycb")
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
set(CMAKE_VERBOSE_MAKEFILE ON)

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
else(NOT OpenCV_FOUND)
	message(STATUS "OpenCV Include Directory: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "OpenCV Link Libraries: ${OpenCV_LIBS}")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS} "/home/yuxiang/Softwares/mesa-17.0.0-rc1/include")

include_directories(${EIGEN3_INCLUDE_DIR}
                    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

link_directories("/home/yuxiang/Softwares/mesa-17.0.0-rc1/lib")

link_libraries(${OpenCV_LIBS}
               assimp
               util
               OSMesa)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;")

cuda_add_library(
  render
  SHARED
  rendering.cpp
)

cuda_add_executable(rendering
                    rendering.cpp)
