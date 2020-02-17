

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


##################
# BUILD C EXAMPLES
##################

set (example TetrahedraLoop)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(TetrahedraLoop)
compile_cl(VertexGather)
compile_cl(Parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/TetrahedraLoop.h
               ${CMAKE_CURRENT_BINARY_DIR}/VertexGather.h
               ${CMAKE_CURRENT_BINARY_DIR}/Parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
