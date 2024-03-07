//
// Created by Andrew Quintana on 2/7/24.
//

#ifndef JETBOTPARKING_DELIVERYMAIN_H
#define JETBOTPARKING_DELIVERYMAIN_H

#include "AprilTagSensor.h"
#include "Astar.h"
#include "ComputationalGeometry.h"
#include "Mapper.h"
#include "DeliveryFSM.h"
#include "OnlineSLAM.h"
#include "RobotSim.h"
#include "Utilities.h"

#include <map>
#include <queue>

// ---------------------------- OBJECT DECLARATION ----------------------------
// state machine
DeliveryFSM fsm = *new DeliveryFSM();
MachineState machine_state = fsm.get_machine_state();

// apriltag
AprilTagSensor* sensor;

// slam algorithm - onlineslam
OnlineSLAM* slam;

// search algorithm - astar
Astar* nav;
state robot_state;
Astar::action next_move;
int beam_resolution = 5;
float max_distance = 0.010; // TODO H2: consider changing max move distance to the minimum scan distance
float heuristic_weight = 1;
float cost = 1;
float fos = 1.25;
float robot_radius = 0.015;


// mapper
std::string window_name = "Radar";
Mapping* mapping;

// robot AND/OR simulator
RobotSim* robot;
float velocity_m_s = 5;
float angular_velocity_rad_s = 5;



// ---------------------------- VARIABLE DECLARATION ----------------------------
int goal_idx = 0;
int goal_ids[] = {1, 2};    // TODO need method of iterating through goals and passing/updating
state goal_state;
env measurements;
std::vector<std::string> obstacle_ids = {"8"};
std::map<std::string, std::vector<int>> obstacle_v_idx;
std::map<std::string, std::vector<int>> obstacle_pov_idx;
std::vector<std::vector<state>> obstacles;
std::priority_queue<std::pair<float,int>> sorted_povs;

// computational geometry
float obstacle_side_m = 0.020;

// SCAN variables
int scan_resolution = 8;
float angle_interval_rad = (2 * M_PI) / scan_resolution;

// ---------------------------- FUNCTION DECLARATION ----------------------------
INFO robot_measure(  );
void robot_rotate( float rotation);
void robot_translate( float translation );
void set_goal_state( state& tag_loc );
void process_obstacles( std::vector<std::vector<state>>& obstacle_states_set );

#endif //JETBOTPARKING_DELIVERYMAIN_H
