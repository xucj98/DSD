cmake_minimum_required(VERSION 3.5)
project(FPGA_TVL1_algorithm)

set(CMAKE_CXX_STANDARD 11)

#add_subdirectory(thirdparty/opencv)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( FPGA_TVL1_algorithm main.cpp BMP.cpp Mat.cpp)

target_link_libraries( FPGA_TVL1_algorithm ${OpenCV_LIBS} )
