cmake_minimum_required(VERSION 2.8)

add_library(reseeds SeedsRevised.cpp Tools.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(reseeds ${OpenCV_LIBS})
