set (example P2Quality)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(quality)
compile_cl(parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/quality.h
            ${CMAKE_CURRENT_BINARY_DIR}/parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
