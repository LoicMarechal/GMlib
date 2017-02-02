
##########################
# BUILD THE STATIC LIBRARY
##########################

add_library(LP.3 lplib3.c)
install (TARGETS LP.3 DESTINATION lib COMPONENT libraries)
install (FILES lplib3.h lplib3.ins DESTINATION include COMPONENT headers)
