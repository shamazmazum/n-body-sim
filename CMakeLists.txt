cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0048 NEW)

project (n-body-sim VERSION 0.1)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")
set (CMAKE_C_FLAGS_RELEASE "-O3")

find_package (OpenCL REQUIRED)
find_package (ProgramMap REQUIRED)

add_subdirectory (src)
add_subdirectory (examples)
