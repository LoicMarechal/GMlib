set(example heat_cell_centered)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
compile_cl(flux)
compile_cl(param)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/flux.h
               ${CMAKE_CURRENT_BINARY_DIR}/param.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)

