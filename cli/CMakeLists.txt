include_directories(../lib/)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

add_executable(cli main.cpp)
target_link_libraries(cli ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} seeds_revised)
