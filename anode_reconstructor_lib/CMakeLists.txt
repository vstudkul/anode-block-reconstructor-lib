cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

project(anode_reconstructor_lib)
 
find_package(PCL 1.2 REQUIRED)
find_package(CGAL REQUIRED)
find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
 
set(CPP_FILES 
        src/load_point_cloud.cpp
        src/reconstruction.cpp
        src/reconstruction_utils.cpp
        src/anode_block_reconstructor.cpp
        )

add_library(anode_reconstructor_lib ${CPP_FILES})
target_link_libraries(anode_reconstructor_lib PUBLIC Open3D::Open3D ${PCL_LIBRARIES} CGAL::CGAL ${OpenCV_LIBS})
