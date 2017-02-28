
################################
# GLOBAL CONFIGURATION VARIABLES
################################

cmake_minimum_required (VERSION 3.0.1)
set (GMlib_VERSION_MAJOR 2)
set (GMlib_VERSION_MINOR 0)
project (GMlib VERSION ${GMlib_VERSION_MAJOR}.${GMlib_VERSION_MINOR} LANGUAGES C)
set (CMAKE_C_FLAGS "-O3")
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set (CMAKE_INSTALL_PREFIX "/opt/GMlib" CACHE PATH "Default GMlib install path" FORCE)
endif()


#######################################
# SET FILES AND DIRECTORIES TO BE BUILT
#######################################

include_directories ("${PROJECT_SOURCE_DIR}/sources")
add_subdirectory (sources)

find_library(OPENCL_LIB OpenCL)
if (OPENCL_LIB)
   include_directories ( ${OPENCL_INCLUDE_DIR} )
else()
   message ("Please set OpenCL library path like this:")
   message ("cmake -DOPENCL_LIB=/cm/shared/apps/cuda70/toolkit/current/lib64/libOpenCL.so")
   message (FATAL_ERROR "OpenCL libary could not be found.")
endif()

find_library (MESHB_LIB Meshb.7 PATHS /opt/libMeshb/lib)
find_library (MESHBF_LIB Meshbf.7 PATHS /opt/libMeshb/lib)
find_path (MESHB_INC libmeshb7.h PATHS /opt/libMeshb/include)

if(MESHB_LIB AND MESHBF_LIB AND MESHB_INC)
   include_directories ("${PROJECT_SOURCE_DIR}/examples")
   include_directories ("${MESHB_INC}")
   add_subdirectory (utilities)
   add_subdirectory (examples)
else()
   message (WARNING "
   libMeshb.7 could not be found.
   Consequently, examples and utilities will not be built.
   You can download it from GitHUB at https://github.com/LoicMarechal/libMeshb")
endif()

install (FILES LICENSE copyright.txt DESTINATION share/GMlib)
install (DIRECTORY sample_meshes DESTINATION share/GMlib)


######################################
# SET PACKAGE AND DEPLOYMENT VARIABLES
######################################

include (InstallRequiredSystemLibraries)
set (CPACK_GENERATOR "TGZ")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copyright.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${GMlib_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${GMlib_VERSION_MINOR}")
set(CPACK_COMPONENTS_ALL applications examples libraries headers)
include (CPack)
