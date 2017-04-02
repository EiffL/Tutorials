cmake_minimum_required(VERSION 2.8)


project(tutorial)

# Find required packages
find_package(Boost COMPONENTS python)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# Build the tools module
add_library(mylib SHARED mylib.cpp)
target_link_libraries(mylib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(mylib PROPERTIES SUFFIX .so)
set_target_properties(mylib PROPERTIES PREFIX "")

