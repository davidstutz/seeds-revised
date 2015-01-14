cmake_minimum_required(VERSION 2.8)

add_library(seeds_revised SeedsRevised.cpp Tools.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(seeds_revised ${OpenCV_LIBS})
