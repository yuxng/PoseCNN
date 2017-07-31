# - Find Nlopt
# Intel Threading Building Blocks offers a rich and complete approach to expressing parallelism in a C++ program
# www.threadingbuildingblocks.org
#
# The module defines the following variables:
# NLOPT_FOUND - the system has nlopt
# NLOPT_INCLUDE_DIR - where to find nlopt.h
# NLOPT_INCLUDE_DIRS - nlopt includes
# NLOPT_LIBRARY - where to find the nlopt library
# NLOPT_LIBRARIES - aditional libraries
# NLOPT_ROOT_DIR - root dir (ex. /usr/local)

# set NLOPT_INCLUDE_DIR
find_path ( NLOPT_INCLUDE_DIR
  NAMES
    nlopt.h
  DOC
    "Nlopt include directory"
)

# set NLOPT_INCLUDE_DIRS
set ( NLOPT_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR} )

# set NLOPT_LIBRARY
find_library ( NLOPT_LIBRARY
  NAMES
    nlopt
  DOC
    "Nlopt library location"
)

# set NLOPT_LIBRARIES
set ( NLOPT_LIBRARIES ${NLOPT_LIBRARY} )

# root dir
# try to guess root dir from include dir
if ( NLOPT_INCLUDE_DIR )
  string ( REGEX REPLACE "(.*)/include.*" "\\1" NLOPT_ROOT_DIR ${NLOPT_INCLUDE_DIR} )

# try to guess root dir from library dir
elseif ( NLOPT_LIBRARY )
  string ( REGEX REPLACE "(.*)/lib[/|32|64].*" "\\1" NLOPT_ROOT_DIR ${NLOPT_LIBRARY} )
endif ()

# handle REQUIRED and QUIET options
include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( Nlopt DEFAULT_MSG NLOPT_LIBRARY
  NLOPT_INCLUDE_DIR
  NLOPT_INCLUDE_DIRS
  NLOPT_LIBRARIES
  NLOPT_ROOT_DIR
)


mark_as_advanced (
  NLOPT_LIBRARY
  NLOPT_LIBRARIES
  NLOPT_INCLUDE_DIR
  NLOPT_INCLUDE_DIRS
  NLOPT_ROOT_DIR
  NLOPT_INTERFACE_VERSION
)
