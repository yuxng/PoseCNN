#pragma once


#ifdef BUILD_DOUBLE

#define DOUBLE_EXPLICIT_INSTANTIATION(macro)  \
    macro(double);

#define DOUBLE_EXPLICIT_INSTANTIATION_WITH_CAMERA_TYPE(macro, type)  \
    macro(double,type);

#else

#define DOUBLE_EXPLICIT_INSTANTIATION(macro)

#define DOUBLE_EXPLICIT_INSTANTIATION_WITH_CAMERA_TYPE(macro, type)

#endif // BUILD_DOUBLE



#ifdef BUILD_POLY3

#include <df/camera/poly3.h>
#define POLY3_EXPLICIT_INSTANTIATION(macro)                     \
    DOUBLE_EXPLICIT_INSTANTIATION_WITH_CAMERA_TYPE(macro,Poly3) \
    macro(float,Poly3)

#else

#define POLY3_EXPLICIT_INSTANTIATION(macro)

#endif // BUILD_POLY3

#ifdef BUILD_LINEAR

#include <df/camera/linear.h>
#define LINEAR_EXPLICIT_INSTANTIATION(macro)                     \
    DOUBLE_EXPLICIT_INSTANTIATION_WITH_CAMERA_TYPE(macro,Linear) \
    macro(float,Linear)

#else

#define LINEAR_EXPLICIT_INSTANTIATION(macro)

#endif // BUILD_LINEAR



#define ALL_CAMERAS_AND_TYPES_INSTANTIATION(macro) \
    POLY3_EXPLICIT_INSTANTIATION(macro);           \
    LINEAR_EXPLICIT_INSTANTIATION(macro)


#define ALL_TYPES_INSTANTIATION(macro)    \
     DOUBLE_EXPLICIT_INSTANTIATION(macro) \
     macro(float)

