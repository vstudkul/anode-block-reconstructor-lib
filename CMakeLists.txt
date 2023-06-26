cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=mold")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

project(anode_reconstructor_project)
 
find_package(PCL 1.2 REQUIRED) 
find_package(Open3D REQUIRED)
add_subdirectory(anode_reconstructor_lib)

add_executable(anode_reconstructor_project main.cpp)
target_link_libraries(anode_reconstructor_project PRIVATE anode_reconstructor_lib Open3D::Open3D ${PCL_LIBRARIES} )
