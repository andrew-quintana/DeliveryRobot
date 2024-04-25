# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.1)

# Project's name, version, and language
project(mylib VERSION 1.0 LANGUAGES CXX)

# Set the include directories
set(EXTERNAL_LIB_DIR "/usr/local/lib")
set(EXTERNAL_INCLUDE_DIR "/usr/local/include")

# Find the AprilTag library
add_library(apriltag SHARED IMPORTED GLOBAL)
set_target_properties(apriltag PROPERTIES
    IMPORTED_LOCATION "${EXTERNAL_LIB_DIR}/libapriltag.so"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include/")

# Find the OpenCV libraries
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if(OpenCV_FOUND)
  include_directories("/usr/local/include/opencv4")
endif()

# Find the Boost library
find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})

# Find the GTest library
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


# Add source files to create a shared library named 'mylib'
add_library(mylib SHARED
  tests/python_cpp_calls/mylib.cpp
  tests/python_cpp_calls/mylib.h
)

# Set properties for the 'mylib' target
set_target_properties(mylib PROPERTIES
  # Set the version of the library to the project's version
  VERSION ${PROJECT_VERSION}
  # Specify the public header file
  PUBLIC_HEADER tests/python_cpp_calls/mylib.h
)

# Link libraries to the shared library
target_link_libraries(mylib
  PRIVATE
    apriltag
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
	${GTEST_LIBRARIES}
	${GTEST_MAIN_LIBRARIES}
)

# Find the Eigen library
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
  target_link_libraries(mylib PUBLIC Eigen3::Eigen)
endif()

# Install the 'mylib' library
install(TARGETS mylib
  # Specify the destination for the library file
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # Specify the destination for the public header file
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the 'mylib.pc' file
install(FILES ${CMAKE_BINARY_DIR}/mylib.pc 
  # Specify the destination for the 'mylib.pc' file
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig
)








