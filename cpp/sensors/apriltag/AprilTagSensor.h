//
// Created by Andrew Quintana on 10/18/23.
//

#ifndef JETBOTPARKING_APRILTAGSENSOR_H
#define JETBOTPARKING_APRILTAGSENSOR_H

#include "Utilities.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

#include "Eigen/Dense"

#include "apriltag/tag16h5.h"
#include "apriltag/apriltag.h"
#include "apriltag/apriltag_pose.h"

#include "time.h"
#include <stdlib.h>

extern std::string calibration_filepath;
extern const char* cal_DIR;

class AprilTagSensor : public Component {
public:
    // constructor & destructor
    AprilTagSensor( const std::string& DIR );
    ~AprilTagSensor();


    // calibration setup
    cv::Mat camera_matrix;
    cv::Mat distance_coefficients;
    cv::Mat rotation_vectors;
    cv::Mat translation_vectors;
    cv::Size frame_size;

    // sensor functions
    bool detect(cv::String image, env& measurements );

private:

    // apriltag variables
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_family_t *tf = tag16h5_create();

    void rotation_conversion(apriltag_pose_t pose, float& roll, float& pitch, float& yaw);
    void print_state(env& states );
};


#endif //JETBOTPARKING_APRILTAGSENSOR_H
