//
// Created by Andrew Quintana on 12/6/23.
//

#ifndef JETBOTPARKING_MAPPER_H
#define JETBOTPARKING_MAPPER_H

#include "Utilities.h"

#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "Eigen/Dense"

using namespace cv;

class Mapping {
private:

    void robot_plotter(Mat& img, state robotState, int scale );
    void apriltag_plotter(Mat& img, std::string id, state landmarkState, int scale );
    void grid_plotter(Mat& image, int spacing, int thickness );
    void obstacle_plotter( Mat& img, std::string id, state obstacleState, int scale );


public:
    Mapping( std::string *window_name );
    ~Mapping();

    void plot_radar(env& states, cv::string savePath, bool close = 1);

    std::string *window_name;
};

void coord_text(Mat& image, int color[3], std::string id, float c_x, float c_y, float psi, int scale );




#endif //JETBOTPARKING_MAPPER_H
