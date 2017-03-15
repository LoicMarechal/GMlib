
##########################
# BUILD THE STATIC LIBRARY
##########################

include_directories (${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/reduce.h
   COMMAND cl2h ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cl ${CMAKE_CURRENT_BINARY_DIR}/reduce.h reduce
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cl)
add_library(GM.2 gmlib2.c reduce.h)
install (FILES gmlib2.h DESTINATION include COMPONENT headers)
install (TARGETS GM.2 EXPORT GMlib-target DESTINATION lib COMPONENT libraries)
install (EXPORT GMlib-target DESTINATION lib/cmake/${PROJECT_NAME})
