//
// Created by Andrew Quintana on 10/18/23.
//

#include "AprilTagSensor.h"
#include "CameraCalibration.h"
#include "Mapper.h"
#include "Utilities.h"

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>

#include "apriltag/common/getopt.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/pjpeg.h"
#include "apriltag/common/zarray.h"
#include "apriltag/common/matd.h"

// ---------------------------- CLASS SETTINGS ----------------------------

// ---------------------------- CLASS DATA TYPES ----------------------------


// ---------------------------- CLASS VARIABLES AND OBJECTS ----------------------------
std::string calibration_filepath = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/camera_calibration";
const char* cal_DIR = calibration_filepath.c_str();


// ---------------------------- GETTERS & SETTERS ----------------------------


// ------------------------- CONSTRUCTOR & DESTRUCTOR -------------------------
AprilTagSensor::AprilTagSensor( const std::string& DIR )
{
    this->debug = false;
    this->verbose = false;

    std::printf("AprilTag sensor setup START.\n");
    // executing primary steps in calibration main function for setup
    const std::string inputSettingsFile = DIR + "/default.xml";
    std::ifstream file(inputSettingsFile);

    if (file.is_open())
    {
        cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings

        // intrinsic parameters
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distance_coefficients;
        fs["rotation_vectors"] >> rotation_vectors;
        fs["translation_vectors"] >> translation_vectors;
        fs["frame_size"] >> frame_size;
        fs.release();
    }
    else
    {
        calibrate_fisheye_checkerboard(DIR);
    }

    apriltag_detector_add_family(td, tf);

    // april tag configuration setup
    std::printf("AprilTag sensor setup COMPLETE.\n");
}

AprilTagSensor::~AprilTagSensor()
{
    apriltag_detector_destroy(td);
    tag16h5_destroy(tf);
}

// ----------------------------- CLASS FUNCTIONS -----------------------------
bool AprilTagSensor::detect(cv::String image, env& measurements )
{
    //
    //std::printf("Preparing image for detection...\n");
    // convert to sensor filetype
    const char *path = image.c_str();

    image_u8_t *im = NULL;
    if (str_ends_with(path, "pnm") || str_ends_with(path, "PNM") ||
        str_ends_with(path, "pgm") || str_ends_with(path, "PGM"))
        im = image_u8_create_from_pnm(path);
    else if (str_ends_with(path, "jpg") || str_ends_with(path, "JPG")) {
        int err = 0;
        pjpeg_t *pjpeg = pjpeg_create_from_file(path, 0, &err);
        if (pjpeg == NULL) {
            printf("pjpeg failed to load: %s, error %d\n", path, err);
            return false;
        }

        im = pjpeg_to_u8_baseline(pjpeg);

        pjpeg_destroy(pjpeg);
    }

    // detecting AprilTags in image
    //std::printf("Setup detection...\n");

    td->quad_decimate = 2.0;    // double:  decimate input image by this factor
    td->quad_sigma = 0.0;       // double:  apply low-pas blur to input; negative sharpens
    td->nthreads = 1;           // int:     insert this many CPU threads
    td->debug = 1;              // bool:    enable debugging output (slow)
    td->refine_edges = 1;       // bool:    spend more time trying to align edges of tags

    //std::printf("Begin detection...\n");
    zarray_t *detections = apriltag_detector_detect(td, im);
    std::cout << "FOUND " << zarray_size(detections) << " DETECTIONS" << std::endl;

    //std::printf("\t Looping through each AprilTag\n");
    for (int i = 0; i < zarray_size(detections); i++) {

        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        std::printf("detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
               i, det->family->nbits, det->family->h, det->id, det->hamming, det->decision_margin);

        // check if invalid apriltag detected
        if (std::stoi(std::to_string(det->id)) > 10) {
            std::cout << "WARNING: INVALID APRILTAG DETECTED...";
            continue;
        }
        // pose info object
        // detect tags in image
        apriltag_detection_info_t info;
        info.tagsize = 0.015;

        info.fx = camera_matrix.at<double>(0, 0);
        info.fy = camera_matrix.at<double>(1, 1);
        info.cx = frame_size.width/2;
        info.cy = frame_size.height/2;
        //info.cx = camera_matrix.at<double>(0,2);
        //info.cy = camera_matrix.at<double>(1,2);
        info.det = det;

        apriltag_pose_t pose;

        //std::printf("Getting the pose...\n");
        double err = estimate_tag_pose(&info, &pose);

        float roll, pitch, yaw;
        rotation_conversion(pose, roll, pitch, yaw);
        float r_state[3] = {roll, pitch, yaw};

        std::cout << "APRILTAG ID: " << det->id << std::endl;;

        if (std::abs(pose.t->data[2]) > 1) {
            std::cout << "WARNING: MEASUREMENT OUSTSIDE OF WORKING RANGE.:" << std::endl;
            continue; // consider turning into error
        }
        int psi = 2;
        std::cout << "Assigning pose z: " << pose.t->data[2] * 100 << " angle psi: " << r_state[psi] << std::endl;
        std::cout << "Calculated x: " << pose.t->data[2] * std::sin(r_state[psi]) * 100 << " | Calculated y: " << pose.t->data[2] * std::cos(r_state[psi]) * 100 << std::endl << std::endl;

        for (int j = 0; j < 6; ++j) {
            if (j < 3) {
                std::cout << pose.t->data[j] << std::endl;
            }
            else {
                std::cout << r_state[j-3] << std::endl;
            }
        }

        // correct measuements that have the apriltag facing away
        if (abs(r_state[psi]) < M_PI/2) { r_state[psi] = (r_state[psi] > 0) ? r_state[psi] + M_PI : r_state[psi] - M_PI; }

        // convert to x, y
        float x = pose.t->data[2] * std::sin(r_state[psi])* -500;
        float y = pose.t->data[2] * std::cos(r_state[psi]) * -100;

        // assign calculated and measured values to
        measurements[std::to_string(det->id)] = state(x, y, r_state[psi]);

    }
    return true;
}

void AprilTagSensor::rotation_conversion(apriltag_pose_t pose, float& roll, float& pitch, float& yaw) {

    // create Eigen matrix to use eulerAngles()
    Eigen::Matrix<float, 3, 3> m;

    for (int i = 0; i < pose.R->ncols; ++i) {
        for (int j = 0; j < pose.R->nrows; ++j) {
            int idx = i * pose.R->ncols + j;
            m(i,j) = pose.R->data[idx];
        }
    }

    // calculate eulerAngles()
    Eigen::Vector3f ea = m.eulerAngles(0,1,1);

    // update roll pitch yaw
    roll = ea(0);
    pitch = ea(1);
    yaw = ea(2);
}

void AprilTagSensor::print_state(env& states ) {
    for (auto it = states.begin(); it != states.end(); ++it) {
        std::printf("Pose translation:\n");
        std::cout << it->second(0) << " ";
        std::cout << it->second(1) << " ";
        std::cout << it->second(2)<< std::endl << std::endl;

        std::printf("Printing pose rotation:\n");
        std::cout << it->second(3) << " ";
        std::cout << it->second(4) << " ";
        std::cout << it->second(5) << std::endl << std::endl;
    }
}


/*int main()
{
    AprilTagSensor* sensor = new AprilTagSensor(cal_DIR);

    cv::Mat distorted, frame, gray;

    // get list of images with path
    const std::string ISOSCELES_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/multiple_aprils/isosceles_set";
    const std::string NX_DY_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/pose_validation/0x_dy";
    const std::string DIRECT_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/slam/straight_angle";
    const std::string CX_CY_DIR = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/pose_validation/cx_cy";
    std::list<string> DIRS = {NX_DY_DIR, CX_CY_DIR, ISOSCELES_DIR, DIRECT_DIR};

    for (auto const tag_DIR : DIRS) {

        std::vector<cv::String> images;
        cv::glob(tag_DIR, images);

        //std::cout << "Post Optimize: " << sensor->camera_matrix << std::endl;
        for(int i = 0; i<images.size(); i++) {
            std::cout << images[i] << std::endl;
            //std::printf("AprilTag Detection STARTING for %s.\n", images[i].c_str());
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
            if (sensor->detect(images[i], states, -999) == INFO::NA) {
                continue;
            }

            // make a center state to represent the robot in the middle of a radar
            states["ROBOT"] = { 0, 0, 0};

            // create an instance of Mapping
            std::string windowName = "Radar";
            Mapping mapping(&windowName);

            // plot the radar
            std::string savePath = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/mapping/";
            mapping.plot_radar( states, savePath, 0);
            cv::Mat view = cv::imread(images[i]);
            std::string viewWindow = images[i].substr(images[i].rfind("/") + 1, -4);
            cv::imshow(viewWindow, view);
            cv::resizeWindow(viewWindow, 224, 224);
            cv::moveWindow( viewWindow, 750, 0);
            cv::waitKey(0);
            cv::destroyAllWindows();

            std::cout << images[i] << std::endl;
            std::printf("%s\n\n", images[i].substr(130, -4).c_str());
        }
    }


    return 0;
}*/
