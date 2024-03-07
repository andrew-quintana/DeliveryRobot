//
// Created by Andrew Quintana on 2/5/24.
//

#ifndef JETBOTPARKING_COMPUTATIONALGEOMETRY_H
#define JETBOTPARKING_COMPUTATIONALGEOMETRY_H

#include "Utilities.h"

#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include <map>
#include "Eigen/Dense"

extern bool cg_debug;
extern bool cg_verbose;

float euclidian( state& state1, state& state2 );

bool collinear(state& a, state& b, state& c);

bool left(state& a, state& b, state& c);

bool left_on(state& a, state& b, state& c);

bool intersects(state& a, state& b, state& c, state& d);

std::vector<state> generate_obstacle(state& tag_state, float side_length_m );

std::vector<state> generate_visibility_points(float agent_radius_m, float fos, std::vector<state> obstacle );

state visibility_point(float agent_diameter, float fos, state& a, state& b, state& c);

float distance_to_line_segment( state& c, state& a, state& b );

bool node_test( state& node, std::vector<std::vector<state>> obstacles, float agent_radius_m, float fos );

bool edge_test( state& line_start, state& line_end, std::vector<std::vector<state>> obstacles, float agent_radius_m, float fos );

#endif //JETBOTPARKING_COMPUTATIONALGEOMETRY_H
