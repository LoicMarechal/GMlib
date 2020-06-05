set (example TestFeatures)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(double_precision)
compile_cl(downlink)
compile_cl(neighbours)
compile_cl(uplink)
compile_cl(parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/double_precision.h
               ${CMAKE_CURRENT_BINARY_DIR}/downlink.h
               ${CMAKE_CURRENT_BINARY_DIR}/neighbours.h
               ${CMAKE_CURRENT_BINARY_DIR}/uplink.h
               ${CMAKE_CURRENT_BINARY_DIR}/parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
