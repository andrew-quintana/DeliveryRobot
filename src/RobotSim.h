//
// Created by Andrew Quintana on 1/18/24.
//

#ifndef JETBOTPARKING_ROBOTSIM_H
#define JETBOTPARKING_ROBOTSIM_H

#include "Utilities.h"

#include "Eigen/Dense"

class RobotSim {
private:

    // --------------------------------------- PARAMETERS ---------------------------------------
    float default_distance_m;
    float velocity_m_s;
    float angular_velocity_rad_s;

    // ---------------------------------- STRUCTURES AND TYPES ----------------------------------


    // --------------------------------------- FUNCTIONS ---------------------------------------


public:
    // -------------------------------- CONSTRUCTOR/DECONSTRUCTOR --------------------------------
    RobotSim( float default_distance_m, float velocity_m_s, float angular_velocity_rad_s );
    ~RobotSim();

    // ------------------------------------- GETTERS/SETTERS ------------------------------------
    float get_angular_velocity();
    void set_angular_velocity( float angular_velocity_rad_s );
    float get_velocity();
    void set_velocity( float velocity_m_s );

    // --------------------------------------- FUNCTIONS ---------------------------------------
    /*
     * Use case: Make an "actual" goal with the previous goal, assuming that the currently
     *          measured goal based on sensor+SLAM localization isn't exact. This would
     *          assume movement with an error similar to that of Gaussian noise. The goal
     *          is to better test the ability of the OnlineSLAM by providing an "actual" value
     *          each time.
     */
    static void normalize_angle( float& angle );
    state rotate( float angle, state& robot );
    state translate( float distance, state& robot );


};


#endif //JETBOTPARKING_ROBOTSIM_H
