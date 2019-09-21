set (n_body_sim_program_path ${CMAKE_INSTALL_PREFIX}/share/n-body-sim)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/program_loc.h.in ${CMAKE_CURRENT_BINARY_DIR}/program_loc.h)
include_directories (${CMAKE_CURRENT_BINARY_DIR} ${PROGRAM_MAP_INCLUDE_DIR} ${OpenCL_INCLUDE_DIRS})

add_executable (n-body-sim clstate.c main.c)
target_link_libraries (n-body-sim ${OpenCL_LIBRARIES} ${PROGRAM_MAP_LIBRARY} m)

install (TARGETS n-body-sim RUNTIME
  DESTINATION bin)
install (FILES process.cl DESTINATION share/n-body-sim)
