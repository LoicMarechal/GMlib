
################################
# GLOBAL CONFIGURATION VARIABLES
################################

cmake_minimum_required (VERSION 3.0.1)
set (GMlib_VERSION_MAJOR 2)
set (GMlib_VERSION_MINOR 0)
project (GMlib VERSION ${GMlib_VERSION_MAJOR}.${GMlib_VERSION_MINOR} LANGUAGES C)
set (CMAKE_C_FLAGS "-O3")


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

find_package(Threads)
find_package(libMeshb 7)

if(libMeshb_INCLUDE_DIR)
   include_directories ("${libMeshb_INCLUDE_DIR}")
   include_directories ("${PROJECT_SOURCE_DIR}/examples")
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


######################################
# EXPORT CMAKE PACKAGE INFORMATION
######################################

include(CMakePackageConfigHelpers)
file(WRITE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
"set(GMlib_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(GMlib_LIBRARY ${CMAKE_INSTALL_PREFIX}/lib/GMlib.${GMlib_VERSION_MAJOR}.a)
")
write_basic_package_version_file(
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
   VERSION ${PROJECT_VERSION}
   COMPATIBILITY SameMajorVersion)
install(FILES
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
   DESTINATION lib/cmake/${PROJECT_NAME})
