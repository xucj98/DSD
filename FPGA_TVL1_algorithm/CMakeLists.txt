cmake_minimum_required(VERSION 3.12)
project(FPGA_TVL1_algorithm)

set(CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/opencv2/include)

#add_subdirectory(thirdparty/opencv)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( FPGA_TVL1_algorithm main.cpp )

target_link_libraries( FPGA_TVL1_algorithm ${OpenCV_LIBS} )
