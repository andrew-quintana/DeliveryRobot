#ifndef MYLIB_H
#define MYLIB_H

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include "opencv2/opencv.hpp"

#include <eigen3/Eigen/Dense>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <gtest/gtest.h>

#ifdef __cplusplus
extern "C" {
#endif


void testApriltag();

void testOpencv( const char* DIR);

bool testEigen();

bool testBoost();

void testGTest();

// calibration setup
cv::Mat camera_matrix;
cv::Mat distance_coefficients;
cv::Mat rotation_vectors;
cv::Mat translation_vectors;
cv::Size frame_size;

#ifdef __cplusplus
}
#endif

#endif // MYLIB_H
