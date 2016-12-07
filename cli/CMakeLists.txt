include_directories(../lib/)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

add_executable(reseeds_cli main.cpp)
target_link_libraries(reseeds_cli ${Boost_LIBRARIES} ${OpenCV_LIBS} reseeds)
