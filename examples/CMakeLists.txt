
###############################################################
# MACRO TO CREATE INTERMEDIARY HEADER FILES FROM OPENCL SOURCES
# BEFORE COMPILING THE EXECUTABLE
###############################################################

macro(compile_cl_example basename)

include_directories (${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h
   COMMAND cl2h ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h ${basename}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl)
add_executable(${basename} ${basename}.c ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h)
target_link_libraries(${basename} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${basename} DESTINATION share/GMlib/examples COMPONENT examples)

endmacro(compile_cl_example)


##################
# BUILD C EXAMPLES
##################

compile_cl_example(test)
