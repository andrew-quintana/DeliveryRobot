# Minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.1)

# Project's name, version, and language
project(delivery_interface VERSION 1.0 LANGUAGES CXX)

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

# Add source files to create a shared library named 'delivery_interface'
add_library(delivery_interface SHARED
  tests/python_cpp_calls/DeliveryFSM.cpp
  tests/python_cpp_calls/delivery_interface.h
)

# Set properties for the 'delivery_interface' target
set_target_properties(delivery_interface PROPERTIES
  # Set the version of the library to the project's version
  VERSION ${PROJECT_VERSION}
  # Specify the public header file
  PUBLIC_HEADER tests/python_cpp_calls/delivery_interface.h
)

# Link libraries to the shared library
target_link_libraries(delivery_interface
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
  target_link_libraries(delivery_interface PUBLIC Eigen3::Eigen)
endif()

# Install the 'delivery_interface' library
install(TARGETS delivery_interface
  # Specify the destination for the library file
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  # Specify the destination for the public header file
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the 'delivery_interface.pc' file
install(FILES ${CMAKE_BINARY_DIR}/delivery_interface.pc 
  # Specify the destination for the 'delivery_interface.pc' file
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig
)








