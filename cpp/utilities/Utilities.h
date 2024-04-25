//
// Created by Andrew Quintana on 1/7/24.
//

#ifndef JETBOTPARKING_UTILITIES_H
#define JETBOTPARKING_UTILITIES_H

#include "DeliveryFSM.h"

#include <iostream>
#include <cstdio>
#include <string>
#include <iomanip>
#include "Eigen/Core"
#include "Eigen/Dense"
#include <map>
#include <random>

class Component {
public:
    Component();
    ~Component();

    bool logging;
    bool debug;
    bool verbose;
};

typedef Eigen::Vector3f state;
typedef std::map<std::string, state > env;

float deg_rad( float deg );

void print_matrix( std::string matrix_name, const Eigen::MatrixXf& matrix );

void insert_rows_and_cols(Eigen::MatrixXf& matrix, int n, int m, int r, int c);

void print_state(env& states );

bool approximatelyEqual(float a, float b, float tolerance = 0.001); // default tolerance at 1 mm

float compute_cartesian();
float compute_bearing();
float truncate_angle();

float random_gauss( float relative, float stddev = 1 );


// print functions

std::string format_string(const char* format, ...);

std::string timestamp();

void status_update( int hyphen_sets, const char* format, ...);

template<typename... Args>
void time_print( const std::string& format, Args... args ) {
    std::string str = format_string(format, args...);
    std::printf("\n%s %s\n",
                timestamp().c_str(),
                str.c_str());
}



#endif //JETBOTPARKING_UTILITIES_H
