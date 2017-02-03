
################################
# GLOBAL CONFIGURATION VARIABLES
################################

cmake_minimum_required (VERSION 3.0.1)
set (GMlib_VERSION_MAJOR 2)
set (GMlib_VERSION_MINOR 0)
project (GMlib  VERSION ${GMlib_VERSION_MAJOR}.${GMlib_VERSION_MINOR} LANGUAGES C)
set (CMAKE_C_FLAGS "-O3")
find_library (MESHB_LIB Meshb.7 PATHS /opt/libMeshb/lib)
find_library (MESHBF_LIB Meshbf.7 PATHS /opt/libMeshb/lib)
find_library(OPENCL_LIB OpenCL)
find_path (MESHB_INC libmeshb7.h PATHS /opt/libMeshb/include)
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   set (CMAKE_INSTALL_PREFIX "/opt/GMlib" CACHE PATH "Default GMlib install path" FORCE)
endif()


#######################################
# SET FILES AND DIRECTORIES TO BE BUILT
#######################################

include_directories ("${PROJECT_SOURCE_DIR}/sources")
include_directories ("${PROJECT_SOURCE_DIR}/examples")
include_directories ("${MESHB_INC}")
include_directories ( ${OPENCL_INCLUDE_DIR} )
add_subdirectory (utilities)
add_subdirectory (sources)
add_subdirectory (examples)
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
