
################################
# GLOBAL CONFIGURATION VARIABLES
################################

cmake_minimum_required (VERSION 3.7.2)
set (GMlib_VERSION_MAJOR 3)
set (GMlib_VERSION_MINOR 30)
project (GMlib VERSION ${GMlib_VERSION_MAJOR}.${GMlib_VERSION_MINOR} LANGUAGES C)

option(WITH_CPACK "Enable cpack target to generate a zip file containing binaries" OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

if (CMAKE_HOST_SYSTEM_NAME STREQUAL Linux)
   set (LINK_LIBRARIES "${LINK_LIBRARIES}m")
endif ()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
   if (WIN32)
      set(CMAKE_INSTALL_PREFIX "$ENV{HOMEPATH}/cmakebuilds" CACHE PATH "..." FORCE)
   else ()
      set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/cmakebuilds/$ENV{OSTYPE}-$ENV{MACHTYPE}" CACHE PATH "..." FORCE)
   endif()
endif ()

list (APPEND CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}")


#####################################################
# CUSTOM COMMAND TO CREATE INTERMEDIARY HEADER FILES 
# FROM OPENCL SOURCES BEFORE COMPILING THE EXECUTABLE
#####################################################

macro(compile_cl basename)

add_custom_command (
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h
   COMMAND cl2h ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h ${basename}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl )

endmacro(compile_cl)


#######################################
# SET FILES AND DIRECTORIES TO BE BUILT
#######################################

find_package(OpenCL)

if (OpenCL_FOUND)
   include_directories (${OpenCL_INCLUDE_DIRS})
else()
   message ("OpenCL could not be found, please set it with:")
   message ("cmake -DOpenCL_LIBRARIES=PATH_TO_OPENCL_LIB/libOpenCL.so -DOpenCL_INCLUDE_DIRS=PATH_TO_OPENCL_INC/include")
   message (FATAL_ERROR "OpenCL libary could not be found.")
endif()

include_directories (${PROJECT_SOURCE_DIR}/sources)

find_package(libMeshb 7.5)

if(libMeshb_FOUND)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DWITH_LIBMESHB")
   include_directories (${libMeshb_INCLUDE_DIRS})
   include_directories (${PROJECT_SOURCE_DIR}/examples)
   add_subdirectory (utilities)
   add_subdirectory (examples/BallQuality)
   add_subdirectory (examples/HeatCellCentered)
   add_subdirectory (examples/MeshQuality)
   add_subdirectory (examples/NodeSmoother)
   add_subdirectory (examples/P2Quality)
   add_subdirectory (examples/TestFeatures)
   message (STATUS "LibMeshb include directory: " ${libMeshb_INCLUDE_DIRS})
else()
   message (WARNING "
   libMeshb 7.5 could not be found (or you may have an older version installed).
   Consequently, examples and utilities will not be built.
   You can download it from GitHub at https://github.com/LoicMarechal/libMeshb")
endif()

add_subdirectory (sources)

install (FILES LICENSE.txt copyright.txt DESTINATION share/GMlib)
install (DIRECTORY sample_meshes DESTINATION share/GMlib)


######################################
# SET PACKAGE AND DEPLOYMENT VARIABLES
######################################

if (WITH_CPACK)
   include (InstallRequiredSystemLibraries)
   set (CPACK_GENERATOR TXZ)
   set (CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/copyright.txt)
   set (CPACK_PACKAGE_VERSION_MAJOR ${GMlib_VERSION_MAJOR})
   set (CPACK_PACKAGE_VERSION_MINOR ${GMlib_VERSION_MINOR})
   set(CPACK_COMPONENTS_ALL applications examples libraries headers)
   include (CPack)
endif ()


######################################
# EXPORT CMAKE PACKAGE INFORMATION
######################################

include(CMakePackageConfigHelpers)
file(WRITE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
"include(\${CMAKE_CURRENT_LIST_DIR}/GMlib-target.cmake)
find_package(OpenCL)
set(GMlib_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include)
set(GMlib_LIBRARIES GM.3 ${OpenCL_LIBRARIES})
set(GMlib_FOUND TRUE)
")
write_basic_package_version_file(
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
   VERSION ${PROJECT_VERSION}
   COMPATIBILITY SameMajorVersion)
install(FILES
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
   DESTINATION lib/cmake/${PROJECT_NAME})


##################################
# PRINT OPERATIONS TO BE CONDUCTED
##################################

message("-- Build mode            : " ${CMAKE_BUILD_TYPE})
message("-- cpack target enabled  : " ${WITH_CPACK})
message("-- Install directory     : " ${CMAKE_INSTALL_PREFIX})
