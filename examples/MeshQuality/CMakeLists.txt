set (example MeshQuality)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(compute_quality)
compile_cl(parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/compute_quality.h
               ${CMAKE_CURRENT_BINARY_DIR}/parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
