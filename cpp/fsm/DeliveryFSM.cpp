//
// Created by Andrew Quintana on 2/6/24.
//

#include "JETBOTPARKING_DELIVERYINTERFACE_H.h"
#include "delivery_interface.h"


DeliveryFSM::DeliveryFSM() {}
DeliveryFSM::~DeliveryFSM() {}

void DeliveryFSM::command_next( INFO next ) {
    // function to determine the next state based on the current state and information produced

    std::string input;

    if (next == INFO::NA) { std::printf("\tNO COMMAND PROVIDED\t"); return; }

    switch (machine_state) {
        case MachineState::INITIALIZE:
            switch (next) {
                case INFO::NA:
                    machine_state = MachineState::SCAN;
                    break;
                case INFO::ERROR:
                    machine_state = MachineState::ERROR;
                    break;
                default:
                    std::printf("\tCOMMAND NOT RECOGNIZED IN INITIALIZE STATE\t");
                    break;
            }

        case MachineState::SCAN:
            // scan FSM
            switch (scan_state) {
                case ScanState::ROTATE:
                    switch (next) {
                        case INFO::GOAL_FOUND:
                            machine_state = MachineState::APPROACH;
                            break;
                        case INFO::GOAL_NOT_FOUND:
                            scan_state = ScanState::POV_MOVE;
                            break;
                        case INFO::AT_GOAL:
                            machine_state = MachineState::PARKING;
                            break;
                        case INFO::ERROR:
                            machine_state = MachineState::ERROR;
                            scan_state = ScanState::ERROR;
                            break;
                        default:
                            std::printf("\tCOMMAND NOT RECOGNIZED IN ROTATE STATE\t");
                            break;
                    }
                    break;

                case ScanState::POV_MOVE:
                    switch (next) {
                        case INFO::NOT_AT_GOAL:
                            scan_state = ScanState::POV_MOVE;
                            break;
                        case INFO::GOAL_FOUND:
                            machine_state = MachineState::APPROACH;
                            break;
                        case INFO::GOAL_NOT_FOUND:
                            scan_state = ScanState::ROTATE;
                            break;
                        case INFO::AT_GOAL:
                            machine_state = MachineState::PARKING;
                            break;
                        case INFO::ERROR:
                            machine_state = MachineState::ERROR;
                            scan_state = ScanState::ERROR;
                            break;
                        default:
                            std::printf("\tCOMMAND NOT RECOGNIZED IN POV_MOVE STATE\t");
                            break;
                    }
                    break;
                default:
                    std::printf("\tSTATE INVALID\t");
                    break;
            }

        case MachineState::APPROACH:
            switch (next) {
                case INFO::AT_GOAL:
                    machine_state = MachineState::PARKING;
                    break;
                case INFO::NOT_AT_GOAL:
                    break;
                case INFO::GOAL_NOT_FOUND:
                    machine_state = MachineState::SCAN;
                    scan_state = ScanState::ROTATE;
                    break;
                case INFO::ERROR:
                    machine_state = MachineState::ERROR;
                    approach_state = ApproachState::ERROR;
                    break;
                default:
                    std::printf("\n\tCOMMAND NOT RECOGNIZED IN APPROACH STATE\t");
                    break;
            }
            break;

        case MachineState::PARKING:
            switch(next) {
                case INFO::NA:
                    machine_state = MachineState::PARKING;
                case INFO::GOAL_FOUND:
                    machine_state = MachineState::APPROACH;
                    break;
                case INFO::GOAL_NOT_FOUND:
                    machine_state = MachineState::SCAN;
                    break;
                case INFO::AT_GOAL:
                    std::cout << "Repeat process? (y/n): ";
                    std::cin >> input;
                    if (input == "y") {
                        machine_state = MachineState::INITIALIZE;
                        break;
                    } else if (input == "n") {

                    }
                    break;
                case INFO::ERROR:
                    machine_state = MachineState::ERROR;
                    parking_state = ParkingState::ERROR;
                default:
                    std::printf("\n\tCOMMAND NOT RECOGNIZED IN APPROACH STATE\t");
                    break;
            }
    }
}

MachineState DeliveryFSM::get_machine_state() { return machine_state; }

ScanState DeliveryFSM::get_scan_state() { return scan_state; }

ApproachState DeliveryFSM::get_appraoch_state() { return approach_state; }

ParkingState DeliveryFSM::get_parking_state() { return parking_state; }

struct DeliveryFSM::status DeliveryFSM::get_status() {

    DeliveryFSM::status current_status{};
    current_status.machine_state = this->machine_state;
    current_status.scan_state = this->scan_state;
    current_status.approach_state = this->approach_state;
    current_status.parking_state = this->parking_state;

    return current_status;

}