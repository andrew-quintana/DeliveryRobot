//
// Created by Andrew Quintana on 1/7/24.
//

#include "gtest/gtest.h"

#include "AprilTagSensor.h"
#include "Mapper.h"
#include "OnlineSLAM.h"
#include "Astar.h"
#include "Utilities.h"
#include "ComputationalGeometry.h"

#include "gtest/gtest.h"
#include "Eigen/Dense"
#include <map>
#include <iostream>
#include <filesystem>

std::pair<float, float> extract_tag_state( std::string filename ) {
    // Find the position of the underscores
    size_t underscorePos = filename.find('_');

    // Extract the substring before the underscore
    std::string xMeasurement = filename.substr(1, underscorePos - 1);

    // Find the position of the dot
    size_t dotPos = filename.find('.');

    // Extract the substring between the underscore and the dot
    std::string yMeasurement = filename.substr(underscorePos + 2, dotPos - underscorePos - 2);

    // Convert the extracted substrings to integers
    std::pair<float, float> output = std::make_pair(std::stof(xMeasurement),std::stof(yMeasurement));

    return output;
}

bool isWithinTolerance(float value, float target, float tolerance) {
    return std::abs(value - target) <= tolerance;
}

class APRILTAGSENSOR : public testing::Test {
protected:
    bool debug_sensor = true;
    bool verbose_sensor = false;

    // list of image dirs
    std::string CAL_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/camera_calibration";
    const std::string ISOSCELES_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/multiple_aprils/isosceles_set";
    const std::string NX_DY_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/pose_validation/0x_dy";
    const std::string DIRECT_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/slam/straight_angle";
    const std::string CX_CY_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/pose_validation/cx_cy";
    std::list<string> DIRS = {DIRECT_DIR, NX_DY_DIR, CX_CY_DIR, ISOSCELES_DIR};
    std::map<std::string,std::vector<std::pair<float, float>>> expected_states;
    std::map<std::string,std::vector<std::pair<float, float>>> measured_states;

    // declare sensor object
    AprilTagSensor* sensor;

    void SetUp() override {
        // instantiate sensor object
        sensor = new AprilTagSensor(CAL_DIR);

        cv::Mat distorted, frame, gray;

        // iterating through each of the DIRs
        for (const auto& dir : DIRS) {
            // setup image glob
            std::vector<cv::String> images;
            cv::glob(dir, images);

            // create new vector of state dictionaries
            expected_states[dir] = std::vector<std::pair<float, float>>();
            measured_states[dir] = std::vector<std::pair<float, float>>();

            // iterate through images in the glob
            for(int i = 0; i<images.size(); i++) {
                // get expected state and image
                expected_states[dir].push_back(extract_tag_state(images[i]));
                distorted = cv::imread(images[i]);

                // failed import check
                if (distorted.empty()) {
                    std::cout << "\033[33m" << "Warning 100: Invalid Filetype - Image file " << i
                              << " wasn't able to be imported." << std::endl << "\033[0m";
                    continue;
                }

                // create map that contains landmark locations
                env states;

                // detect landmark locations in view
                INFO detection_result = sensor->detect(images[i], states, -1);
                if (detection_result == INFO::GOAL_FOUND) {
                    std::printf("GOAL FOUND");
                }
                if (detection_result == INFO::NA) {
                    continue;
                }

                for (const auto& loc : states) {
                    measured_states[dir].push_back(std::make_pair(loc.second[0],loc.second[1]));
                }

                // make a center state to represent the robot in the middle of a radar
                states["ROBOT"] = { 0, 0, 0};

                // create an instance of Mapping
                if (debug_sensor) {
                    std::string windowName = "Radar";
                    Mapping mapping(&windowName);

                    // plot the radar
                    std::string savePath = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/mapping/";
                    mapping.plot_radar(states, savePath, 0);
                    cv::Mat view = cv::imread(images[i]);
                    std::string viewWindow = images[i].substr(images[i].rfind("/") + 1, -4);
                    cv::imshow(viewWindow, view);
                    cv::resizeWindow(viewWindow, 224, 224);
                    cv::moveWindow(viewWindow, 750, 0);
                    cv::waitKey(0);
                    cv::destroyAllWindows();

                    std::cout << images[i] << std::endl;
                    std::printf("%s\n\n", images[i].substr(130, -4).c_str());
                }

            }
        }

    }
};

TEST_F(APRILTAGSENSOR, Calibration) {

    const std::string expected_calibration_file = CAL_DIR + "/default.xml";
    bool file_exists = std::filesystem::exists(expected_calibration_file);
    ASSERT_TRUE(file_exists);

}

TEST_F(APRILTAGSENSOR, Measurements) {

    for (auto const& test_dir : DIRS) {
        for (int i = 0; i < expected_states[test_dir].size(); ++i) {
            float expected_x = expected_states[test_dir][i].first;
            float expected_y = expected_states[test_dir][i].first;
            float measured_x = measured_states[test_dir][i].second;
            float measured_y = measured_states[test_dir][i].second;

            ASSERT_TRUE(isWithinTolerance(expected_x, measured_x, 0.010));
            ASSERT_TRUE(isWithinTolerance(expected_y, measured_y, 0.010));
        }
    }

}

class ONLINESLAM : public testing::Test {
protected:
    // measurements instance
    env measurements;

    void SetUp() override {

        // input measurements
        measurements["1"] = state(60., 70., 1.);
        measurements["2"] = state(-50., 30., -2.);
        measurements["3"] = state(-100., 20., -4);
        measurements["8"] = state(20., 100., -3.);

    }

    // SLAM Instance
    int rows = measurements.begin()->second.rows();

};

// -------------------------------- TESTING OBJECTS --------------------------------


// -------------------------------- SETUP FUNCTIONS --------------------------------


// -------------------------------- UNIT TESTS --------------------------------

TEST_F(ONLINESLAM, ProcessMeasurements) {

    Eigen::MatrixXf sol_Xi(15,1);
    sol_Xi <<   70,
            -220,
            8,
            60,
            70,
            1,
            -50,
            30,
            -2,
            -100,
            20,
            -4,
            20,
            100,
            -3;

    // assertions for test
    OnlineSLAM measTest = OnlineSLAM(rows);
    ASSERT_TRUE(measTest.process_measurements( measurements ));
    ASSERT_EQ(measTest.get_Xi(), sol_Xi);
    for (int i = 0; i < measTest.get_Xi().rows(); i++) {
        for (int j = 0; j < measTest.get_Xi().cols(); j++) {
            if (measTest.get_Xi()(i, j) != sol_Xi(i, j)) {
                std::cout << "Element at position (" << i << ", " << j << ") is different." << std::endl;
                std::cout << measTest.get_Xi()(i, j) << " NEQ " << sol_Xi(i, j) << std::endl;
                std::cout << std::boolalpha << (measTest.get_Xi()(i, j) == sol_Xi(i, j)) << std::endl;
                std::cout << measTest.get_Xi()(i, j) - sol_Xi(i, j) << std::endl;
            }
        }
    }
}

TEST_F(ONLINESLAM, ProcessMovement) {

    OnlineSLAM moveTest = OnlineSLAM(rows);
    moveTest.process_measurements( measurements );
    ASSERT_TRUE(moveTest.process_movement( 5, M_PI/4 ));


}


TEST_F(ONLINESLAM, MapUpdate) {

    OnlineSLAM mapTest = OnlineSLAM(rows);
    mapTest.process_measurements( measurements );
    mapTest.process_movement( 5, M_PI/4 );
    ASSERT_TRUE(mapTest.map_update());

}

class ASTAR : public testing::Test {
protected:

    Astar* robot_nav;
    state robot_state;

    void SetUp() override {
        cg_debug = true;
        cg_verbose = false;

        robot_nav = new Astar( 5, 0.010, 1,
                               1, 1, 0.015);

        robot_state = state(0.0f, 0.0f, 0.0f);

    }

    void TearDown() override {
        delete robot_nav;
    }

};

TEST_F(ASTAR, InPlaneMove) {

    // straight line to goal
    state goal_state = state(0.1, 0, 0);
    robot_nav->set_goal(goal_state);

    // run beam search
    std::vector<std::vector<state>> obstacles;
    Astar::action test = robot_nav->astar_move(robot_state, obstacles, goal_state);
    ASSERT_TRUE(test.next == INFO::GOAL_FOUND);
}

TEST_F(ASTAR, OrthogonalMove) {

    // orthogonal move to goal
    state goal_state = state(0,0.1,0);
    robot_nav->set_goal(goal_state);

    // run beam search
    std::vector<std::vector<state>> obstacles;
    Astar::action test = robot_nav->astar_move(robot_state, obstacles, goal_state);
    ASSERT_TRUE(test.next == INFO::GOAL_FOUND);
}

TEST_F(ASTAR, AvoidObstacle) {

    // straight line to goal
    state goal_state = state(0.2,0,0);
    robot_nav->set_goal(goal_state);

    // run beam search with obstacle directly in front
    std::vector<std::vector<state>> obstacles =
            {
                    {state( 0.025, 0.025, 0),
                     state( 0.025, -0.025, 0),
                     state( 0.050, -0.025, 0),
                     state( 0.050, 0.025, 0)}
            };
    Astar::action test = robot_nav->astar_move(robot_state, obstacles, goal_state);
    ASSERT_TRUE(test.next == INFO::GOAL_FOUND);
}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

