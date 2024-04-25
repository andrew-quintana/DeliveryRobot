//
// Created by Andrew Quintana on 12/6/23.
// Based on implementation: https://docs.opencv.org/3.4/d3/d96/tutorial_basic_geometric_drawing.html
//

#include "Mapper.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <format>
#include <vector>
#include <map>

Mapping::Mapping( std::string *window_name ) {
    this->window_name = window_name;

};

Mapping::~Mapping() {}

void coord_text(Mat& image, int color[3], std::string id, float c_x, float c_y, float psi, int scale ) {

    int img_cx = image.cols/2;
    int img_cy = image.rows/2;

    // Define the text to be displayed
    std::string text = format("%s (%0.3f,%0.3f,%0.3f)", id.c_str(), c_x, -c_y, psi);

    // Define the font properties
    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    float fontScale = scale;
    cv::Scalar fontColor(color[0], color[1], color[2]); // blue color
    int thickness = 2;

    // Calculate the position to place the text
    int offset = (c_x > 0) ? 10 : -325;
    if (c_x == 0) { offset = 10; }
    cv::Point textPosition(img_cx + c_x + offset, img_cy + c_y);

    // Put the text on the image
    cv::putText(image, text, textPosition, fontFace, fontScale, fontColor, thickness);
}

void Mapping::robot_plotter(Mat& img, state robotState, int scale ) {

    // define local variables
    int lineType = 8;
    int img_cx = img.cols/2;
    int img_cy = img.rows/2;
    float c_x = -robotState[0];
    float c_y = -robotState[1];
    float psi = robotState[2];
    int color[3] = {0,0, 255};

    // create shape coordinates and rotate to desired position
    Eigen::MatrixXf shapePoints(3,2);
    shapePoints <<    0, -3,
                      2,  2,
                     -2,  2;
    float angle = deg_rad(psi);
    Eigen::Rotation2D<float> rotation(angle);
    shapePoints = shapePoints * rotation.matrix();
    shapePoints *= scale;

    Point robotPoints[1][3];
    for (int i = 0; i < shapePoints.rows(); ++i) {
        robotPoints[0][i] = Point( img_cx + c_x + shapePoints(i,0), img_cy + c_y + shapePoints(i,1));
    }

    const Point* ppt[1] = { robotPoints[0] };
    int npt[] = { 3 };

    fillPoly( img,
              ppt,
              npt,
              1,
              Scalar(color[0], color[1], color[2]), // blue
              lineType);

    coord_text(img, color, "R", c_x, c_y, psi, 1);
}



void Mapping::apriltag_plotter( Mat& img, std::string id, state landmarkState, int scale ) {

    // define local variables
    int lineType = 8;
    int img_cx = img.cols/2;
    int img_cy = img.rows/2;
    float c_x = -landmarkState[0];
    float c_y = -landmarkState[1];
    float psi = landmarkState[2];
    int color[3] = {255,0,0};

    Point tagPoints[1][3];
    Eigen::MatrixXf shapePoints(3,2);
    shapePoints <<    0, -3,
                      2,  2,
                     -2,  2;

    //float angle = deg_rad(psi);
    Eigen::Rotation2D<float> rotation(psi);
    shapePoints = shapePoints * rotation.matrix();
    shapePoints *= scale;

    for (int i = 0; i < shapePoints.rows(); ++i) {
        tagPoints[0][i] = Point( img_cx + c_x + shapePoints(i,0), img_cy + c_y + shapePoints(i,1));
    }
    const Point* ppt[1] = { tagPoints[0] };
    int npt[] = { 3 };

    fillPoly( img,
              ppt,
              npt,
              1,
              Scalar(color[0], color[1], color[2]), // red
              lineType);

    coord_text(img, color, id, c_x, c_y, psi, 1);

}

void Mapping::obstacle_plotter( Mat& img, std::string id, state obstacleState, int scale ) {
    // define local variables
    int lineType = 8;
    int img_cx = img.cols/2;
    int img_cy = img.rows/2;
    float c_x = -obstacleState[0];
    float c_y = -obstacleState[1];
    float psi = obstacleState[2];
    int color[3] = {255,255,255};   // black

    Point tagPoints[1][3];
    Eigen::MatrixXf shapePoints(4,2);
    shapePoints <<      -1, -2,
                         1, -2,
                         1,  0,
                        -1,  0;

    //float angle = deg_rad(psi);
    Eigen::Rotation2D<float> rotation(psi);
    shapePoints = shapePoints * rotation.matrix();
    shapePoints *= scale;

    for (int i = 0; i < shapePoints.rows(); ++i) {
        tagPoints[0][i] = Point( img_cx + c_x + shapePoints(i,0), img_cy + c_y + shapePoints(i,1));
    }
    const Point* ppt[1] = { tagPoints[0] };
    int npt[] = { 3 };

    fillPoly( img,
              ppt,
              npt,
              1,
              Scalar(color[0], color[1], color[2]),
              lineType);

    coord_text(img, color, id, c_x, c_y, psi, 1);
}

void Mapping::grid_plotter(Mat& image, int spacing, int thickness ) {
    int rows = image.rows / spacing;
    int cols = image.cols / spacing;
    int opacity = 10;
    int colorValue = 255 * opacity/100;
    int color[3] = {colorValue,colorValue,colorValue};

    for (int i = 0; i < rows; i++) {
        line(image, Point(0, i * spacing), Point(image.cols, i * spacing), Scalar(color[0], color[1], color[2]), thickness);
    }

    for (int j = 0; j < cols; j++) {
        line(image, Point(j * spacing, 0), Point(j * spacing, image.rows), Scalar(color[0], color[1], color[2]), thickness);
    }
}

void Mapping::plot_radar(env& states, cv::string savePath, bool close) {

    std::string testName;

    // create Matd
    int rows = 900;
    int cols = 700;
    int color[3] = {255,255,255};
    cv::Mat plot(rows, cols, CV_8UC3, cv::Scalar(color[0], color[1], color[2]));

    // plot grid
    grid_plotter(plot, 10, 1);

    // plot landmarks
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->first == "ROBOT") {
            robot_plotter(plot, it->second, 3);;
        }
        else if (std::stoi(it->first) > 5) {
            obstacle_plotter(plot, it->first, it->second, 3);
        }
        else {
            apriltag_plotter(plot, it->first, it->second, 3);
        }
    }

    imshow( window_name->c_str(), plot);
    moveWindow( window_name->c_str(), 750, 0);

    std::string filename = savePath + "radar_image.jpg";
    imwrite(filename, plot);

    if (close) {
        // wait for a key press
        waitKey(0);
        destroyAllWindows();
    }
}

/*
int main() {
    // create an instance of Mapping
    std::string window_name = "Radar";
    Mapping mapping(&window_name);

    // create some test data
    float robotState[3] = {0, 0, 0};

    std::map<std::string, std::vector<float> > tagStates;
    tagStates["5"].push_back(-50);
    tagStates["5"].push_back(-200);
    tagStates["5"].push_back(180);

    tagStates["1"].push_back(-150);
    tagStates["1"].push_back(-150);
    tagStates["1"].push_back(270);

    tagStates["0"].push_back(100);
    tagStates["0"].push_back(-50);
    tagStates["0"].push_back(90);

    // plot the radar
    std::string savePath = "/Users/aq_home/Library/CloudStorage/OneDrive-Personal/1ODocuments/Projects/jetbot_parking/simulated_testing/mapping/";
    mapping.plot_radar(robotState, tagStates, savePath);

    return 0;
}
*/
