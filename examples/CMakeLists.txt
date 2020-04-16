

##################
# BUILD C EXAMPLES
##################

set (example TF_test_features)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(TF_double_precision)
compile_cl(TF_downlink)
compile_cl(TF_neighbours)
compile_cl(TF_uplink)
compile_cl(Parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/TF_double_precision.h
               ${CMAKE_CURRENT_BINARY_DIR}/TF_downlink.h
               ${CMAKE_CURRENT_BINARY_DIR}/TF_neighbours.h
               ${CMAKE_CURRENT_BINARY_DIR}/TF_uplink.h
               ${CMAKE_CURRENT_BINARY_DIR}/Parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)

set (example MQ_mesh_quality)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(MQ_mesh_quality)
compile_cl(Parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/MQ_mesh_quality.h
            ${CMAKE_CURRENT_BINARY_DIR}/Parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)

set (example NS_nodes_smoothing)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(NS_quality)
compile_cl(NS_scatter)
compile_cl(NS_gather)
compile_cl(Parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/NS_quality.h
               ${CMAKE_CURRENT_BINARY_DIR}/NS_scatter.h
               ${CMAKE_CURRENT_BINARY_DIR}/NS_gather.h
            ${CMAKE_CURRENT_BINARY_DIR}/Parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)

add_subdirectory(heat_cell_centered)
