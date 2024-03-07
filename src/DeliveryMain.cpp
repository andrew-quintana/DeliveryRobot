//
// Created by Andrew Quintana on 2/6/24.
//

#include "DeliveryMain.h"

INFO robot_measure(  ) {

    // take and process image
    std::string image_filename = sensor->take_image();
    INFO measure_output = sensor->detect(image_filename, measurements, goal_idx);

    // process recognized states and clear
    slam->process_measurements(measurements);
    measurements.clear();

    // process obstacles, updating or adding locations
    process_obstacles(obstacles);

    return measure_output;

}

void robot_rotate( float rotation) {
    // move to new angle
    robot->rotate(rotation, robot_state);

    // process movement
    slam->process_movement(0,rotation);
}

void robot_translate( float translation ) {
    // move to new position
    robot->translate(translation, robot_state);

    // process movement
    slam->process_movement(0,translation);
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