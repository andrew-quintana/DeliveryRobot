cmake_minimum_required(VERSION 3.26)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(${JetbotParking_SOURCE_DIR}/include)
include_directories(${JetbotParking_SOURCE_DIR}/src)
link_directories(${JetbotParking_SOURCE_DIR}/src)

add_executable(JetbotParking
        AprilTagSensor.cpp
        AprilTagSensor.h
        Astar.cpp
        Astar.h
        BoostGraph.cpp
        BoostGraph.h
        CameraCalibration.cpp
        CameraCalibration.h
        ComputationalGeometry.cpp
        ComputationalGeometry.h
        DeliveryFSM.cpp
        DeliveryFSM.h
        DeliveryMain.cpp
        DeliveryMain.h
        ErrorHandling.cpp
        Mapper.cpp
        Mapper.h
        OnlineSLAM.cpp
        OnlineSLAM.h
        RobotSim.cpp
        RobotSim.h
        test.cpp
        Utilities.cpp
        Utilities.h
        #pywrap.cpp
)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(AprilTag)

#find_package(pybind11 REQUIRED)


# Link libraries
find_library(APRILTAG_LIBRARY apriltag PATHS /Users/aq_home/Downloads/JetbotParking/include/lib/)

target_link_libraries(JetbotParking
        PRIVATE Eigen3::Eigen
        ${GTEST_MAIN_LIBS}
        ${GTEST_LIBS}
        ${OpenCV_LIBS}
        ${APRILTAG_LIBRARY}
        #${PYBIND11_LIBS}
)