//
// Created by Andrew Quintana on 1/18/24.
//

#include "RobotSim.h"

// -------------------------------- CONSTRUCTOR/DECONSTRUCTOR --------------------------------

RobotSim::RobotSim( float default_distance_m, float velocity_m_s, float angular_velocity_rad_s ) {

    // set class parameters per inputs
    this->default_distance_m = default_distance_m;
    this->velocity_m_s = velocity_m_s;
    this->angular_velocity_rad_s = angular_velocity_rad_s;

}

// ------------------------------------- GETTERS/SETTERS ------------------------------------

// angular velocity
float RobotSim::get_angular_velocity() {
    return angular_velocity_rad_s;
}

void RobotSim::set_angular_velocity( float angular_velocity_rad_s ) {
    // TODO (H2): Conversion based on angular velocity to tangential velocity equation.
    this->angular_velocity_rad_s = angular_velocity_rad_s;
}

// velocity
float RobotSim::get_velocity() {
    return velocity_m_s;
}

void RobotSim::set_velocity( float velocity_m_s ) {
    this->velocity_m_s = velocity_m_s;
}

// --------------------------------------- CLASS FUNCTIONS ---------------------------------------
void RobotSim::normalize_angle( float& angle ) {

    // determine modulo wrt pi
    int mod = fmod(angle,static_cast<float>(M_PI));

    if ( abs(mod) > 1 ) {
        if (angle > 0) {
            angle -= 2 * M_PI * mod;
        }
        else {
            angle += 2 * M_PI * mod;
        }
    }
}

// --------------------------------------- OBJECT FUNCTIONS ---------------------------------------

state RobotSim::rotate( float angle, state& robot ) {

    state sim_state;

    // TODO (H2): check trig math
    // calculate move and add gaussian noise (noise in distance, x, y)
    sim_state[0] = robot[0] * (1 + random_gauss(robot[0]));
    sim_state[1] = robot[1] * (1 + random_gauss(robot[1]));
    sim_state[2] = robot[2] + angle * (1 + random_gauss(robot[2]));

    return sim_state;

}

state RobotSim::translate( float distance, state& robot ) {

    state sim_state;

    // TODO (P2): check trig math
    // calculate move and add gaussian noise (noise in distance, delta_x, delta_y)
    float delta_d = distance * (1 + random_gauss(distance));        // simmed distance traveled

    sim_state[0] = robot[0] + (delta_d) * std::sin(robot[2]);           // simmed x value
    sim_state[1] = robot[1] + (delta_d) * std::cos(robot[2]);           // simmed y value
    sim_state[2] = robot[2] * (1 + random_gauss(robot[2]));         // simmed resultant angle


    return sim_state;
}

float random_gauss( float mean, float stddev ) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a normal distribution
    std::normal_distribution<float> dist(mean, stddev);

    return dist(gen);
}