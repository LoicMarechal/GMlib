
#################
# BUILD UTILITIES
#################

add_executable(hilbert hilbert.c)
target_link_libraries(hilbert LP.3 ${MESHB_LIB})
install (TARGETS hilbert DESTINATION bin COMPONENT applications)
