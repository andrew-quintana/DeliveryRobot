//
// Created by Andrew Quintana on 12/6/23.
//

#include "OnlineSLAM.h"

// -------------------------------- CONSTRUCTOR/DESTRUCTOR --------------------------------

OnlineSLAM::OnlineSLAM( int dim ) :
        // initialize estimate matrix
        mu(dim,1),
        Omega(dim, dim),
        Xi(dim,1) {
    /*
     * Description
     * -----------
     * Constructor of the Online SLAM algorithm
     *
     * Arguments
     * ---------
     *      - dim: the dimensions tracked and updated by the algorithm
     */

    if (debug) { std::cout << std::endl << ("----------------- START SLAM CONSTRUCTOR -----------------") << std::endl << std::endl; }

    // dimensions of system
    this->dim = dim;

    // initialize omega and xi matrices
    Eigen::MatrixXf identity = Eigen::MatrixXf::Identity(dim, dim);
    Omega = identity;
    Xi.setZero();

    // set initial robot location to center and facing "north"
    states["ROBOT"].setZero();

    if (verbose) { print_matrix( "Omega", Omega); }
    if (verbose) { print_matrix( "Xi", Xi); }
    if (debug) { std::cout << std::endl << ("----------------- END SLAM CONSTRUCTOR -----------------") << std::endl << std::endl; }


}

OnlineSLAM::~OnlineSLAM() {
    // Implement the destruction logic if needed
}

// -------------------------------- GETTERS & SETTERS --------------------------------

env OnlineSLAM::getMap() {
    /*
     * Description
     * -----------
     * Process a new series of measurements and update (x,y) location of robot and landmarks
     *
     * Output
     * ---------
     *      - map in the form of {landmark ID : state matrix}
     */

    return states;
}

// Getter for Xi
Eigen::MatrixXf OnlineSLAM::get_Xi() {
    return Xi;
}

// Getter for Omega
Eigen::MatrixXf OnlineSLAM::get_Omega() {
    return Omega;
}

// ---------------------------------- SLAM FUNCTIONS ----------------------------------

int OnlineSLAM::process_measurements(env measurements ) {
    /*
     * Description
     * -----------
     * Process a new series of measurements and update (x,y) location of robot and landmarks
     *
     * Arguments
     * ---------
     *      - measurements: Collection of measurements of tree positions and radius in the format {landmark ID : state matrix}
     *
     */

    if (debug) { std::cout << std::endl << ("-----------------START MEASUREMENTS PROCESSING-----------------") << std::endl << std::endl; }

    // for every landmark measurement in current dataset
    for (auto it = measurements.begin(); it != measurements.end(); ++it) {

        // do not include ROBOT location
        if (it->first == "ROBOT") {
            continue;
        }

        // coordinate import
        float landmark_bearing_normalized_rad = states["ROBOT"](2) + it->second(2);
        if (debug) {
            std::printf("Pulled landmark information for landmark %s: %f, %f, %f\n",
                        it->first.c_str(),
                        it->second(0),
                        it->second(1),
                        landmark_bearing_normalized_rad);
        }

        // create new landmark if first time seeing
        if (landmarks.count(it->first) == 0) {

            // add to tracked map and landmarks list
            states[it->first] << it->second(0), it->second(1), landmark_bearing_normalized_rad;
            landmarks[it->first] << -1;

            // append rows and columns to Omega
            Eigen::MatrixXf temp_Omega(Omega.rows() + dim, Omega.cols() + dim);
            temp_Omega.topLeftCorner(Omega.rows(), Omega.cols()) = Omega;
            Omega = temp_Omega;
            Eigen::MatrixXf temp_Xi(Xi.rows() + dim, 1);
            temp_Xi.topLeftCorner(Xi.rows(), 1) = Xi;
            Xi = temp_Xi;
            landmarks[it->first] = Omega.rows() - dim;

            if (debug) { std::cout << it->first << " added to list of landmarks." << std::endl;}
        }

        // integrate measurements
        int idx = landmarks[it->first];
        for (int b = 0; b < dim; ++b) {
            Omega(b,b) += 1;
            Omega(idx + b,idx + b) += 1;
            Omega(idx + b,b) += -1;
            Omega(b,idx + b) += -1;
            Xi(b,0) += -it->second[b];
            Xi(idx + b,0) += it->second[b];
        }

        if (verbose) { print_matrix( "Omega", Omega); }
        if (verbose) { print_matrix( "Xi", Xi); }
    }
    if (debug) { std::cout << std::endl << ("-----------------END MEASUREMENTS PROCESSING-----------------") << std::endl << std::endl; }
    return 1;
}

int OnlineSLAM::process_movement(float translation_m, float rotation_rad ) {
    /*
     * Description
     * -----------
     * Process a new series of movement commands and update (x,y) location of robot
     *
     * Arguments
     * ---------
     *      - translation_m: forward distance in meters commanded for the robot
     *      - rotation_rad: rotation_rad in radians commanded for the robot
     *
     */

    if (debug) { std::cout << std::endl << ("-----------------START MOVEMENT PROCESSING-----------------") << std::endl << std::endl; }
    if (verbose) { std::printf("Translation: %f | Rotation: %f\n", translation_m, rotation_rad); }

    // determine new position


    states["ROBOT"](2) += rotation_rad;
    float new_robot_delta_x_m = translation_m * cos(states["ROBOT"][2]);
    float new_robot_delta_y_m = translation_m * sin(states["ROBOT"][2]);
    state estimate = {new_robot_delta_x_m, new_robot_delta_y_m, rotation_rad};
    if (verbose) { std::printf("Requested motion: %f, %f, %f\n", new_robot_delta_x_m, new_robot_delta_y_m, rotation_rad); }

    // expand information matrix and vector by one new position
    // Omega resizing
    insert_rows_and_cols(Omega,dim,dim,dim,dim);

    // Xi resizing
    insert_rows_and_cols(Xi,dim,0,dim,0);

    // update information matrix/vector based on the robot motion
    if (verbose) { print_matrix("Omega pre move update", Omega); }
    for (int b = 0; b < dim * 2; ++b) {
        Omega(b,b) += 1;
    }
    for (int b = 0; b < dim; ++b) {
        Omega(dim + b, b) += -1;
        Omega(b, dim + b) += -1;
        Xi(b, 0) += -estimate(b);
        Xi(dim + b, 0) += estimate(b);
    }

    if (verbose) { print_matrix( "Omega", Omega); }
    if (verbose) { print_matrix( "Xi", Xi); }
    if (debug) { std::cout << ("-----------------END MOVEMENT PROCESSING-----------------") << std::endl; }

    return 1;
}

int OnlineSLAM::map_update() {
    /*
     * Description
     * -----------
     * Update map based on processing of class matrices
     *
     */

    if (debug) { std::cout << std::endl << ("-----------------START MAP UPDATING-----------------") << std::endl << std::endl; }

    if (verbose) { print_matrix( "Omega", Omega); }
    if (verbose) { print_matrix( "Xi", Xi); }

    // determine submatrices for calculation
    Eigen::MatrixXf a;
    a = Omega.block(0, dim,  dim,  Omega.rows() - dim);
    if (verbose) { print_matrix("A", a); }
    Eigen::MatrixXf b;
    b = Omega.block(0, 0, dim, dim);
    if (verbose) { print_matrix("B", b); }
    Eigen::MatrixXf c;
    c = Xi.block(0, 0, dim, 1);
    if (verbose) { print_matrix("C", c); }
    Eigen::MatrixXf pOmega;
    pOmega = Omega.block(dim, dim, Omega.rows()-dim, Omega.cols()-dim);
    if (verbose) { print_matrix("Omega'", pOmega); }
    Eigen::MatrixXf pXi;
    pXi = Xi.block(dim, 0, Xi.rows()-dim, 1);
    if (verbose) { print_matrix("Xi'", pXi); }

    // calculate Omega and Xi
    Omega = pOmega - a.transpose() * b.inverse() * a;
    Xi = pXi - a.transpose() * b.inverse() * c;
    if (verbose) { std::cout << "FACTORED IN PREVIOUS POSE AND CALCULATED MU:" << std::endl; }
    if (verbose) { print_matrix( "Omega", Omega); }
    if (verbose) { print_matrix( "Xi", Xi); }

    // determine mu
    mu = Omega.inverse() * Xi;
    if (verbose) { print_matrix( "Mu", mu); }

    // map update using for loop and auto
    for (auto it = states.begin(); it != states.end(); ++it) {
        int idx = landmarks[it->first];
        for (int i = 0; i < dim; ++i) {
            it->second(i) = mu(idx+i,0);
        }
    }
    if (verbose) { print_state(states); }
    if (debug) { std::cout << std::endl << ("-----------------END MAP UPDATING-----------------") << std::endl << std::endl; }

    return 1;
}