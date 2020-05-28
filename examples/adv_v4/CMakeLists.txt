# C Compiler options. 
set(CMAKE_C_STANDARD 90)
set(CMAKE_C_STANDARD_REQUIRED ON) 
set(CMAKE_C_EXTENSIONS OFF) 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-long-long" CACHE STRING "" FORCE)

set(example adv_v4)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
compile_cl(param)
compile_cl(all_crd)
compile_cl(bal_crd)


add_executable(${example} ${example}.c
               ${CMAKE_CURRENT_BINARY_DIR}/param.h
               ${CMAKE_CURRENT_BINARY_DIR}/all_crd.h
               ${CMAKE_CURRENT_BINARY_DIR}/bal_crd.h
               )
target_link_libraries(${example} GM.3 ${libMeshb_LIBRARIES} ${OpenCL_LIBRARIES})
install (TARGETS ${example} DESTINATION share/GMlib/examples COMPONENT examples)
