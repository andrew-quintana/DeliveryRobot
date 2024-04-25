//
// Created by Andrew Quintana on 2/5/24.
//

#include "ComputationalGeometry.h"

bool cg_debug = true;
bool cg_verbose = false;

// euclidian euclidian function per: http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
float euclidian( state& state1, state& state2 ) {

    float x1_m = state1[0];
    float y1_m = state1[1];
    float x2_m = state2[0];
    float y2_m = state2[1];

    float dx = abs(x1_m - x2_m);
    float dy = abs(y1_m - y2_m);

    float heuristic = sqrt(pow(dx,2) + pow(dy,2));

    return heuristic;

}

// function to check if points a, b, and c are collinear
bool collinear(state& a, state& b, state& c) {
    return (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) == 0;
}

// function to check if point c is on the left of line segment ab
bool left(state& a, state& b, state& c) {
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0;
}

// function to check if point c is on the left of line segment ab or on the line itself
bool left_on(state& a, state& b, state& c) {
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) >= 0;
}

bool intersects(state& a, state& b, state& c, state& d) {

    std::vector<state> point_set = {a, b, c, d};

    // check if any point on any line segment
    if (
            collinear(a, b, c) ||
            collinear(a, b, d) ||
            collinear(c, d, a) ||
            collinear(c, d, b)
            ) return false;

    // check for intersection based on each point's side wrt line segment
    if ( !(
            (left(a, b, c) != left(a, b, d)) &&
            (left(c, d, a) != left(c, d, b))
            )) return false;

    return true;

}

std::vector<state> generate_obstacle(state& tag_state, float side_length_m ) {

    // point in xyz
    state tag_point(tag_state[0], tag_state[1], 0);

    // translation vectors for each point given 0 rad rotation
    state a(0, side_length_m / 2, 0);
    state b(-side_length_m, side_length_m / 2, 0);
    state c(-side_length_m, -side_length_m / 2, 0);
    state d(0, -side_length_m / 2, 0);

    // rotation angle defined as orthogonal direction of face
    float angle = tag_state[2];
    state axis( 0,0,1);                 // rotation about z axis

    // create translation matrix
    Eigen::Translation3f a_translation_matrix(a);
    Eigen::Translation3f b_translation_matrix(b);
    Eigen::Translation3f c_translation_matrix(c);
    Eigen::Translation3f d_translation_matrix(d);

    // create rotation matrix
    Eigen::AngleAxisf rotation(angle, axis);
    Eigen::Matrix3f rotation_matrix = rotation.toRotationMatrix();

    // apply to translation followed by rotation to initial point
    state A = rotation_matrix * (a_translation_matrix * tag_point);
    state B = rotation_matrix * (b_translation_matrix * tag_point);
    state C = rotation_matrix * (c_translation_matrix * tag_point);
    state D = rotation_matrix * (d_translation_matrix * tag_point);

    return std::vector<state>{A, B, C, D};

}

std::vector<state> generate_visibility_points(float agent_radius_m, float fos, std::vector<state> obstacle ) {
    std::vector<state> obstacle_povs = std::vector<state>(obstacle.size());

    for (int j = 0; j < obstacle.size(); ++j) {
        int i = (j == 0) ? i = obstacle.size() - 1 : i - 1;
        int k = (j == obstacle.size() - 1) ? k = 0 : k = j + 1;
        obstacle_povs[j] = visibility_point(agent_radius_m, fos, obstacle[i], obstacle[j], obstacle[k]);
    }

    return obstacle_povs;
}

// function to calculate the visibility point for an agent within a triangle
state visibility_point(float agent_radius, float fos, state& a, state& b, state& c) {
    // states normalized to corner
    float x_b = a[0] - b[0];
    float y_b = a[1] - b[1];
    float x_c = a[0] - c[0];
    float y_c = a[1] - c[1];

    // calculate components
    float theta_b = atan2(y_b,x_b);
    float theta_c = atan2(y_c,x_c);
    float length_visibility = agent_radius * fos;

    // calculate visibility angle
    float phi = (theta_c - theta_b) / 2 + theta_b;

    // calculate visibility point coordinates
    float x_v = a[0] + length_visibility * cos(phi);
    float y_v = a[0] + length_visibility * sin(phi);

    return state(x_v, y_v, 0);

}

float distance_to_line_segment( state& point, state& line_start, state& line_end ) {
    // implementation based on code from Georgia Tech CS7632

    // return minimum distance between line segment and point
    float l2 = (line_end - line_start).squaredNorm();
    state line_start_to_point_m = point - line_start;

    // points are identical and distance can be measured directly
    if (l2 == 0.0) return line_start_to_point_m.norm();

    state line = line_end - line_start;
    float t = fmax(0.0f, fmin(1.0f, line_start_to_point_m.dot(line) / l2));
    state projection = line_start + t * line;

    return euclidian(point, projection);
}

bool node_test( state& node, std::vector<std::vector<state>> obstacles, float agent_radius_m, float fos ) {

    // TODO (P3): determine if node within environment

    // TODO (P3): determine if node too close to boundary of environment

    // iterate through obstacle edges
    for (std::vector<state> obstacle_points : obstacles) {

        int left_count = 0;

        for (int i = 0; i < obstacle_points.size(); ++i) {

            // determine adjacent vertex index with wraparound
            int j = (i == obstacle_points.size() - 1) ? 0 : i + 1;

            if (cg_verbose) std::printf("NODE TO OBSTACLE EDGE DIST = %f < %f:\tNODE: (%f, %f)\tEDGE (%f, %f)-(%f, %f)\n",
                                      distance_to_line_segment(node, obstacle_points[i], obstacle_points[j]),
                                      agent_radius_m * fos,
                                      node[0], node[1],
                                      obstacle_points[i][0],obstacle_points[i][1],
                                      obstacle_points[j][0],obstacle_points[j][1]);

            //determine if too close to obstacle
            if (distance_to_line_segment(node, obstacle_points[i], obstacle_points[j]) < agent_radius_m * fos) {
                if (cg_debug) std::printf("NODE TOO CLOSE TO OBSTACLE EDGE:\tNODE: (%f, %f)\tEDGE (%f, %f)-(%f, %f)\n",
                                       node[0], node[1],
                                       obstacle_points[i][0],obstacle_points[i][1],
                                       obstacle_points[j][0],obstacle_points[j][1]);
                return false;
            }

            // count number of left_on
            if (left_on(node, obstacle_points[i], obstacle_points[j])) left_count += 1;
        }

        // determine if node on/inside of polygon
        if (cg_verbose) { std::printf("LEFT COUNT = %i of %i\n", left_count, obstacle_points.size()); } //TODO REMOVE
        if (left_count == obstacle_points.size()) {
            if (cg_debug) std::printf("NODE INSIDE OBSTACLE:\tNODE: (%f, %f)\tOBSTACLE LEFT_ON %i\n",
                                   node[0], node[1],
                                   left_count);
        }
    }

    return true;

}

bool edge_test( state& line_start, state& line_end, std::vector<std::vector<state>> obstacles, float agent_radius_m, float fos ) {

    for (std::vector<state> obstacle_points : obstacles) {
        for (int i = 0; i < obstacle_points.size(); ++i) {

            // determine adjacent vertex index with wraparound
            int j = (i == obstacle_points.size() - 1) ? 0 : i + 1;

            if (cg_verbose) std::printf("OBSTACLE VERTEX TO EDGE TEST:\tOB VERTEX: (%f, %f)\tEDGE (%f, %f)-(%f, %f)\n",
                                      obstacle_points[i][0], obstacle_points[i][0],
                                      line_start[0], line_start[1],
                                      line_end[0], line_end[1]);

            // check if potential edge is too lcose to specified obstacle point
            if (distance_to_line_segment(obstacle_points[i], line_start, line_end) < agent_radius_m * fos) {
                if (cg_debug) std::printf("OBSTACLE VERTEX TOO CLOSE TO EDGE:\tOB VERTEX: (%f, %f)\tEDGE (%f, %f)-(%f, %f)\n",
                                       obstacle_points[i][0], obstacle_points[i][0],
                                       line_start[0], line_start[1],
                                       line_end[0], line_end[1]);
                return false;
            }

            if (cg_verbose) std::printf("NODE EDGE INTERSECTION TEST:\t"
                                      "NODE EDGE: (%f, %f)-(%f, %f)\t"
                                      "OBS EDGE: (%f, %f)-(%f, %f)\n",
                                      line_start[0], line_start[1],
                                      line_end[0], line_end[1],
                                      obstacle_points[i][0], obstacle_points[i][1],
                                      obstacle_points[j][0], obstacle_points[j][1]);

            // check if intersection occuring between obstacle edge and nodes
            if (intersects(line_start, line_end, obstacle_points[i], obstacle_points[j])) {
                if (cg_debug) std::printf("NODE EDGE INTERSECTS WITH OBSTACLE EDGE:\t"
                                       "NODE EDGE: (%f, %f)-(%f, %f)\t"
                                       "OBS EDGE: (%f, %f)-(%f, %f)\n",
                                       line_start[0], line_start[1],
                                       line_end[0], line_end[1],
                                       obstacle_points[i][0], obstacle_points[i][1],
                                       obstacle_points[j][0], obstacle_points[j][1]);
                return false;
            }

        }
    }

    return true;

}