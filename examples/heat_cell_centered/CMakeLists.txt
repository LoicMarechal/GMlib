# C Compiler options. 
set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_C_EXTENSIONS OFF) 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-long-long" CACHE STRING "" FORCE)

set(example heat_cell_centered)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
compile_cl(ini_tet)
compile_cl(sol_ext)
compile_cl(grd_tet)
compile_cl(grd_ext)
compile_cl(flx_bal)
compile_cl(tim_int)
compile_cl(param)
compile_cl(dt)
add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/ini_tet.h
               ${CMAKE_CURRENT_BINARY_DIR}/sol_ext.h
               ${CMAKE_CURRENT_BINARY_DIR}/grd_tet.h
               ${CMAKE_CURRENT_BINARY_DIR}/grd_ext.h
               ${CMAKE_CURRENT_BINARY_DIR}/flx_bal.h
               ${CMAKE_CURRENT_BINARY_DIR}/tim_int.h
               ${CMAKE_CURRENT_BINARY_DIR}/param.h
               ${CMAKE_CURRENT_BINARY_DIR}/dt.h)
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
