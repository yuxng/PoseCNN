project("synthesizer")
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

include(${CMAKE_SOURCE_DIR}/cmake/FindNLopt.cmake)

find_package(CUDA REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
pkg_check_modules(OpenEXR REQUIRED OpenEXR)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(BOOST_INCLUDE_DIRS "/usr/local/include")
set(BOOST_LIBRARY_DIRS "/usr/local/lib/boost")

set(xFusion_ROOT ${PROJECT_SOURCE_DIR}/../kinect_fusion)
set(xFusion_INCLUDE_DIRS ${xFusion_ROOT}/include)
set(xFusion_LIBRARY_DIRS ${xFusion_ROOT}/build)
set(xFusion_LIBRARIES kfusion)

# TODO:
set(SUITESPARSE_INCLUDE_DIRS "/usr/include/suitesparse" CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIBRARIES "cholmod;metis")

message(STATUS "xFusion root: ${xFusion_ROOT}")
message(STATUS "pangolin: ${Pangolin_INCLUDE_DIRS}")

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

include_directories(${Pangolin_INCLUDE_DIRS}
                    ${xFusion_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${OpenEXR_INCLUDE_DIRS}
                    ${SUITESPARSE_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/include
                    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
                    ${BOOST_INCLUDE_DIRS})

link_directories(${Pangolin_LIBRARY_DIRS}
                 ${xFusion_LIBRARY_DIRS}
                 ${OpenEXR_LIBRARY_DIRS}
                 ${PCL_LIBRARY_DIRS}
                 ${BOOST_LIBRARY_DIRS})

link_libraries(${Pangolin_LIBRARIES}
               ${xFusion_LIBRARIES}
               ${DEPTHSENSE_LIBRARIES}
               ${SUITESPARSE_LIBRARIES}
               ${OpenEXR_LIBRARIES}
               ${OpenCV_LIBS}
               ${NLOPT_LIBRARIES}
               ${PCL_LIBRARIES}
               boost_python
               boost_numpy
               assimp
               util)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;")

cuda_add_library(
  synthesizer
  SHARED
  synthesize.cpp
  thread_rand.cpp
  Hypothesis.cpp
)

#cuda_add_executable(synthesize
#                    synthesize.cpp
#                    thread_rand.cpp
#                    Hypothesis.cpp)
