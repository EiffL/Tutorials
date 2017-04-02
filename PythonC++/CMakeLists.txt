cmake_minimum_required(VERSION 2.8)
include(ExternalProject)

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(tutorial)

# Find required packages
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

# Compile boost packages to match the specific system environment
include(BuildBoost)

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

# Build the tools module
add_library(mylib SHARED mylib.cpp)
target_link_libraries(mylib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(mylib PROPERTIES SUFFIX .so)
set_target_properties(mylib PROPERTIES PREFIX "")

add_library(filter SHARED filterlib.cpp filter.cpp)
target_link_libraries(filter ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(filter PROPERTIES SUFFIX .so)
set_target_properties(filter PROPERTIES PREFIX "")
