//
// Created by Andrew Quintana on 2/6/24.
//

#ifndef JETBOTPARKING_DELIVERYFSM_H
#define JETBOTPARKING_DELIVERYFSM_H

#pragma once

#include "AprilTagSensor.h"
#include "Astar.h"
#include "ComputationalGeometry.h"
#include "Mapper.h"
#include "OnlineSLAM.h"
#include "RobotSim.h"
#include "Utilities.h"

#include <map>
#include <array>
#include <iostream>
#include <queue>

enum class INFO {
    GOAL_FOUND,
    GOAL_NOT_FOUND,
    AT_GOAL,
    NOT_AT_GOAL,
    ERROR,
    NA
};

enum class MachineState {
    INITIALIZE,
    SCAN,
    APPROACH,
    PARKING,
    SHUTDOWN,
    ERROR
};

enum class ScanState {
    ROTATE,
    POV_MOVE,
    ERROR
};

enum class ApproachState {
    MOVE,
    ERROR
};

enum class ParkingState {
    ORTHOGONALIZE,
    REVERSE,
    ERROR
};

class DeliveryFSM : public Component {
private:

    MachineState machine_state = MachineState::INITIALIZE;

    ScanState scan_state = ScanState::ROTATE;

    ApproachState approach_state = ApproachState::MOVE;

    ParkingState parking_state = ParkingState::ORTHOGONALIZE;

    struct status {
        MachineState machine_state;
        ScanState scan_state;
        ApproachState approach_state;
        ParkingState parking_state;
    };


public:

    DeliveryFSM();
    ~DeliveryFSM();

    void command_next(INFO next = INFO::NA );

    MachineState get_machine_state();
    ScanState get_scan_state();
    ApproachState get_appraoch_state();
    ParkingState get_parking_state();
    struct status get_status();


};

std::string step_fsm();


#endif //JETBOTPARKING_DELIVERYFSM_H
