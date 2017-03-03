
###############################################################
# MACRO TO CREATE INTERMEDIARY HEADER FILES FROM OPENCL SOURCES
# BEFORE COMPILING THE EXECUTABLE
###############################################################

macro(compile_cl_example basename)

include_directories ("${CMAKE_CURRENT_BINARY_DIR}")
add_custom_command(
   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h
   COMMAND cl2h ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h ${basename}
   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${basename}.cl)
add_executable(${basename} ${basename}.c ${CMAKE_CURRENT_BINARY_DIR}/${basename}.h)
target_link_libraries(${basename} GM.2 ${MESHB_LIB} ${OPENCL_LIB} ${LINK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS ${basename} DESTINATION share/GMlib/examples COMPONENT examples)

endmacro(compile_cl_example)


##################
# BUILD C EXAMPLES
##################

compile_cl_example(EdgesBasicLoop)
compile_cl_example(EdgesDependenciesLoop)
compile_cl_example(TrianglesBasicLoop)
compile_cl_example(TrianglesDependenciesLoop)
compile_cl_example(QuadsBasicLoop)
compile_cl_example(QuadsDependenciesLoop)
compile_cl_example(TetrahedraBasicLoop)
compile_cl_example(TetrahedraDependenciesLoop)
compile_cl_example(HexahedraBasicLoop)
compile_cl_example(HexahedraDependenciesLoop)
