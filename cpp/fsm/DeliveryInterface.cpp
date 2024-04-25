//
// Created by Andrew Quintana on 2/6/24.
//

#include "DeliveryInterface.h"
#include "fsm.h"

std::string step_fsm( std::string img_path ) {
    // determine INFO enum for updating FSM state and output string for python interpreter

    // every single run should measure from the most recent picture
    if (fsm.get_machine_state != MachineState::INITIALIZE) { measure_output = robot_measure(img_path); }
    if (measure_output == INFO::ERROR) { command_next(measure_output); return "error 0 0"; }

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

            next_info = INFO::NA;
            output = "na 0 0"

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

                    rotate++;   // increment number of rotations

                    // most recent
                    if (measure_output == INFO::GOAL_FOUND) {
                        output = "wait 0 0";
                    }
                    else {
                        output = std::format("move {} {}", 0, angle_interval_rad);
                    }

                    next_info = measure_output;

                    break;

                case ScanState::POV_MOVE:
                    // move toward POV generated around obstacles OR other points in the room
 
                    // check if goal seen in process of navigating POVs
                    if (measure_output == INFO::GOAL_FOUND) {
                        next_info = measure_output;
                        output = "wait 0 0";
                        break;
                    }
                    // check if at goal in process of navigating POVs
                    else if (euclidian(robot_state, goal_state) <= max_distance * fos)
                    {
                        next_info = INFO::AT_GOAL
                        output = "wait 0 0";
                        break;
                    }
                    

                    // determine incremental movement to furthers pov
                    // TODO make this the closest then to rotate around
                    int inspect_pov = sorted_povs.top().second;
                    Astar::action next_move = nav->astar_move(
                            slam->getMap()["ROBOT"],
                            obstacles,
                            nav->get_graph()[inspect_pov].vertex_state);

                    // process info from Astar
                    if (next_move.next == INFO::AT_GOAL)
                    {
                        int num_povs = obstacle_pov_idx.size();
                        povs_visited++;
                        if (povs_visited > num_povs) { }

                        // update to next pov
                        inspect_pov = (inspect_pov == num_povs - 1) ? 0 : inspect_pov + 1;
                        output = "wait 0 0";


                    }
                    // error from astar
                    else if (next_move.next == INFO::ERROR) { output = "error 0 0"; }
                    // output should be INFO::NA to continue in POV state
                    else { output = std::format("move {} {}", next_move.distance, next_move.steering); }

                    next_info = next_move.next;
                
                    break;

            }

        case MachineState::APPROACH:
            // TODO H1: determine if all should be cleaned up to avoid having internal loops like this one

            // determine next step on path
            next_move = nav->astar_move(
                    slam->getMap()["ROBOT"],
                    obstacles,
                    goal_state);

            // error output check
            if (next_move.next == INFO::ERROR) { next_info = next_move.next; output = "error 0 0"; break; }

            // make next move on path
            output = std::format("move {} {}", next_move.distance, next_move.steering);
            next_info = next_move.next;
            break;

        case MachineState::PARKING:
            /*
            * runs three times, once to:
            *       1. move to the delivery location
            *       2. rotate to be orthogonal
            *       3. determine if the next goal has been found
            */

            // approach and reach goal
            if (!parked) {
                output = std::format("move {} {}", euclidian(robot_state, goal_state), goal_state[2] - robot_state[2]);
                parked = true;
                next_info = INFO::NA;
                break;
            }
            
            // rotate to be orthogonal
            if (!orth) {
                output = std::format("move {} {}", euclidian(robot_state, goal_state), goal_state[2] - robot_state[2]);
                orth = true;
                next_info = INFO:NA;
            }

            // update values and prepare for next steps
            parked = false;
            orth = false;
            goal_ids.pop();

            // there are no more goals
            if (goal_ids.empty()) { next_info = INFO::AT_GOAL; output = "done 0 0"; break;}

            // goals left
            goal_idx = goal_ids.front();
            
            // next goal already found
            if (measurements.count(goal_idx) > 0) { next_info = INFO::GOAL_FOUND;}
            else {next_info = INFO::GOAL_NOT_FOUND;}

            output = "wait 0 0";

            break;
    }

    fsm.command_next(next_info);
    return output;

}


INFO robot_measure( std::string img_path ) {

    // process image
    if (!sensor->detect(img_path, measurements, goal_idx)) { return INFO::ERROR; }
    // TODO add way of boxing apriltag and putting the distance on the image

    // process recognized states and clear
    slam->process_measurements(measurements);
    measurements.clear();

    // process obstacles, updating or adding locations
    process_obstacles(obstacles);

    // process last movement
    slam->process_movement(next_move.distance, next_move.steering);

    // check if the goal is a landmark in the environment
    if (measurements.count(goal_idx) > 0) { return INFO::GOAL_FOUND; }
    else { return INFO::GOAL_NOT_FOUND; }

}


void process_obstacles( std::vector<std::vector<state>>& obstacle_states_set ) {

    // iterate through each obstacle
    for (int i = 0; i < obstacle_ids.size(); ++i) {

        // generate obstacle
        state tag_state = slam->getMap()[obstacle_ids[i]];
        std::vector<state> ob = generate_obstacle(tag_state, obstacle_side_m);

        // create povs
        std::vector<state> obstacle_povs = generate_visibility_points(
                robot_radius, fos, ob
        );

        // add new obstacle entry for output
        obstacle_states_set.emplace_back();
        std::vector<state>& obstacle_states = obstacle_states_set.back();

        // add/update obstacle pov points
        for (const auto& id : obstacle_ids) {

            // clear if present to update
            obstacle_v_idx[id].clear();
            obstacle_pov_idx[id].clear();
            obstacle_povs.clear();

            // create Vertex for each obstacle and pov
            for (int j = 0; j < ob.size(); ++j) {

                // obstacle vertex
                int v_idx = nav->create_vertex();
                nav->get_graph()[v_idx].vertex_state = ob[j];
                obstacle_v_idx[id].push_back(v_idx);

                // create new vertex
                int pov_idx = nav->create_vertex();
                nav->get_graph()[pov_idx].vertex_state = ob[j];
                obstacle_v_idx[id].push_back(pov_idx);

                // add to cg interpretable obstacles
                obstacle_states.push_back(ob[j]);

                // add to distance prioritized povs
                float distance = euclidian(slam->getMap()["ROBOT"],ob[j]);
                sorted_povs.push(std::make_pair(distance, pov_idx));

            }

        }

    }

}