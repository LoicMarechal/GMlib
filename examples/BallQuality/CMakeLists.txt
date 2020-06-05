set (example BallQuality)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
compile_cl(triqal)
compile_cl(verqal)
compile_cl(parameters)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/triqal.h
               ${CMAKE_CURRENT_BINARY_DIR}/verqal.h
               ${CMAKE_CURRENT_BINARY_DIR}/parameters.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES} ${LINK_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
