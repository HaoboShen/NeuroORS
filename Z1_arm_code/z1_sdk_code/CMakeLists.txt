cmake_minimum_required(VERSION 3.0)
project(z1_sdk)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -pthread")

include_directories(
  /home/shb/unitree_z1_ws/src/z1_sdk/include
)

link_directories(/home/shb/unitree_z1_ws/src/z1_sdk/lib)
add_executable(z1_test z1_test.cpp)
target_link_libraries(z1_test Z1_SDK_${CMAKE_SYSTEM_PROCESSOR})
