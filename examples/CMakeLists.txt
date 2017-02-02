
##################
# BUILD C EXAMPLES
##################

add_executable(basic_test basic_test.c)
target_link_libraries(basic_test LP.3 ${MESHB_LIB})
install (TARGETS basic_test DESTINATION share/LPlib/examples COMPONENT examples)

add_executable(tetrahedra_neighbours tetrahedra_neighbours.c)
target_link_libraries(tetrahedra_neighbours LP.3 ${MESHB_LIB})
install (TARGETS tetrahedra_neighbours DESTINATION share/LPlib/examples COMPONENT examples)

add_executable(indirect_writes indirect_writes.c)
target_link_libraries(indirect_writes LP.3 ${MESHB_LIB})
install (TARGETS indirect_writes DESTINATION share/LPlib/examples COMPONENT examples)


########################
# BUILD FORTRAN EXAMPLES
########################

enable_language (Fortran)

add_executable(basic_test_f77 basic_test.f)
target_link_libraries(basic_test_f77 LP.3 ${MESHBF_LIB})
install (TARGETS basic_test_f77 DESTINATION share/LPlib/examples COMPONENT examples)

add_executable(indirect_writes_f77 indirect_writes.f)
target_link_libraries(indirect_writes_f77 LP.3 ${MESHBF_LIB})
install (TARGETS indirect_writes_f77 DESTINATION share/LPlib/examples COMPONENT examples)
