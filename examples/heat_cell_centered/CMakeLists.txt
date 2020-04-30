set(example heat_cell_centered)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
compile_cl(ini_tet)
compile_cl(sol_ext)
compile_cl(grd_tet)
compile_cl(grd_ext)
compile_cl(flx_bal)
compile_cl(tim_int)
compile_cl(param)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/ini_tet.h
               ${CMAKE_CURRENT_BINARY_DIR}/sol_ext.h
               ${CMAKE_CURRENT_BINARY_DIR}/grd_tet.h
               ${CMAKE_CURRENT_BINARY_DIR}/grd_ext.h
               ${CMAKE_CURRENT_BINARY_DIR}/flx_bal.h
               ${CMAKE_CURRENT_BINARY_DIR}/tim_int.h
               ${CMAKE_CURRENT_BINARY_DIR}/param.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)

