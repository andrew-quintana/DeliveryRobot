//
// Created by Andrew Quintana on 12/6/23.
//

#ifndef JETBOTPARKING_ONLINESLAM_H
#define JETBOTPARKING_ONLINESLAM_H

#include "AprilTagSensor.h"
#include "Utilities.h"

#include <map>
#include "Eigen/Dense"

class OnlineSLAM : public Component {
public:
    // DEBUGGING AND VISUALIZATION

    // CONSTRUCTOR AND DESTRUCTOR
    OnlineSLAM( int dim );
    ~OnlineSLAM();

    // GETTERS AND SETTERS
    env getMap();
    Eigen::MatrixXf get_Xi();
    Eigen::MatrixXf get_Omega();

    // TEMP WHILE TESTING
    int process_measurements(env measurements );
    int process_movement(float translation_m, float rotation_rad );
    int map_update( );

private:
    // CLASS PARAMETERS
    int dim;        // dimensions of mapped space

    // SLAM matrices
    Eigen::MatrixXf mu;
    Eigen::MatrixXf Omega;
    Eigen::MatrixXf Xi;

    // SLAM FUNCTIONS
    /*int process_measurements( map measurements );
    int process_movement(float translation_m, float rotation_rad );
    int map_update();*/

    // INPUT AND OUTPUT VARIABLES
    env states;  // landmark ID : state matrix
    std::map<std::string, int> landmarks;                       // landmark ID : idx

};

// ---------------------------------- SLAM STRUCTURES ---------------------------------
struct Landmark {
    state loc;
    int id;
};

struct Robot {
    state loc;
};


#endif //JETBOTPARKING_ONLINESLAM_H
