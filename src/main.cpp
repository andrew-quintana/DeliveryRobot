//
// Created by Andrew Quintana on 2/6/24.
//

#include "DeliveryMain.h"

int main(int argc, char** argv) {

    while (machine_state != MachineState::SHUTDOWN){

        switch (machine_state) {
            case MachineState::INITIALIZE:
                // initialize systems
                sensor = new AprilTagSensor(cal_DIR);
                mapping = new Mapping(&window_name);
                slam = new OnlineSLAM(3);
                nav = new Astar( beam_resolution, max_distance, heuristic_weight,
                                 cost, fos, robot_radius);
                robot = new RobotSim(max_distance, velocity_m_s, angular_velocity_rad_s);

                // initialize variables
                goal_state.setZero();

                break;
            case MachineState::SCAN:
                /*
                 * scan has both rotate and pov move for the scalability of scan state.
                 * alternative would be to make pov move the approach step and cache
                 * whether or not the robot was appraoching a tag or a pov, returning to
                 * the scanning step if so.
                 */
                switch (fsm.get_scan_state()) {
                    case ScanState::ROTATE:
                        // rotate at current position to try to identify the location of the target tag

                        INFO scan_output;

                        // rotate completely, measuring the world at each interval
                        for (int i = 0; i < scan_resolution; ++i) {

                            // search via camera
                            scan_output = robot_measure();
                            if (scan_output == INFO::GOAL_FOUND) { fsm.command_next(scan_output); break; }

                            // rotate to continue scan
                            robot_rotate(angle_interval_rad);

                        }

                        // move onto next scanning state if goal not found
                        fsm.command_next(scan_output);
                        break;

                    case ScanState::POV_MOVE:
                        // move toward POV generated around obstacles OR other points in the room

                        while (scan_output == INFO::GOAL_NOT_FOUND) {

                            // determine incremental movement to furthest pov
                            int inspect_pov = sorted_povs.top().second;
                            Astar::action next_move = nav->astar_move(
                                    slam->getMap()["ROBOT"],
                                    obstacles,
                                    nav->get_graph()[inspect_pov].vertex_state);

                            // process info
                            if (next_move.next != INFO::NA) {

                                fsm.command_next(next_move.next);

                            } else {

                                // execute movements
                                robot_rotate(next_move.steering);
                                robot_translate(next_move.distance);

                                // search via camera along the way
                                scan_output = robot_measure();

                            }
                        }

                        fsm.command_next(scan_output);
                        break;

                }

            case MachineState::APPROACH:
                // TODO H1: determine if all should be cleaned up to avoid having internal loops like this one

                // determine next step on path
                next_move = nav->astar_move(
                        slam->getMap()["ROBOT"],
                        obstacles,
                        goal_state);

                // make next move on path
                robot_rotate(next_move.steering);
                robot_translate(next_move.distance);

                // scan to update map
                robot_measure();

                // determine next steps
                fsm.command_next(next_move.next);
                break;

            case MachineState::PARKING:

                // approach and reach goal
                robot_rotate(goal_state[2] - robot_state[2]);
                robot_translate(max_distance);

                // orthogonalize system
                robot_rotate(goal_state[2] - robot_state[2]);

                break;
        }

    }

    status_update(0,"SHUTTING DOWN");


}


