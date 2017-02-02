
################################
# GLOBAL CONFIGURATION VARIABLES
################################

cmake_minimum_required (VERSION 2.6)
project (LPlib)
set (LPlib_VERSION_MAJOR 3)
set (LPlib_VERSION_MINOR 5)
set (CMAKE_C_FLAGS "-O3")
set (CMAKE_Fortran_FLAGS "-O3")
set (CMAKE_INSTALL_PREFIX "/opt/LPlib")
find_library (MESHB_LIB Meshb.7 PATHS /opt/libMeshb/lib)
find_library (MESHBF_LIB Meshbf.7 PATHS /opt/libMeshb/lib)
find_path (MESHB_INC libmeshb7.h PATHS /opt/libMeshb/include)


#######################################
# SET FILES AND DIRECTORIES TO BE BUILT
#######################################

include_directories ("${PROJECT_SOURCE_DIR}/sources")
include_directories ("${MESHB_INC}")
add_subdirectory (sources)
add_subdirectory (examples)
add_subdirectory (utilities)
install (FILES LICENSE copyright.txt DESTINATION share/LPlib)
install (DIRECTORY sample_meshes DESTINATION share/LPlib)


######################################
# SET PACKAGE AND DEPLOYMENT VARIABLES
######################################

include (InstallRequiredSystemLibraries)
set (CPACK_GENERATOR "TGZ")
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copyright.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${LPlib_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${LPlib_VERSION_MINOR}")
set(CPACK_COMPONENTS_ALL applications examples libraries headers)
include (CPack)
