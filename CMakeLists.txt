# setup per https://stackoverflow.com/questions/8304190/cmake-with-include-and-source-paths-basic-setup
cmake_minimum_required(VERSION 3.26)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

project(JetbotParking)

vcpkg_cmake_configure(TARGET JetbotParking)

add_subdirectory(src)

set(CMAKE_CXX_STANDARD 17)
