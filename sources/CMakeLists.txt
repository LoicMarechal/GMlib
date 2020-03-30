
##########################
# BUILD THE STATIC LIBRARY
##########################

include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(reduce)
compile_cl(toolkit)
add_library(GM.3 gmlib3.c ${CMAKE_CURRENT_BINARY_DIR}/reduce.h ${CMAKE_CURRENT_BINARY_DIR}/toolkit.h)
install (FILES gmlib3.h DESTINATION include COMPONENT headers)
install (TARGETS GM.3 EXPORT GMlib-target DESTINATION lib COMPONENT libraries)
install (EXPORT GMlib-target DESTINATION lib/cmake/${PROJECT_NAME})
