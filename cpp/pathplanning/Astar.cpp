//
// Created by Andrew Quintana on 1/16/24.
//

#include "Astar.h"
#include <cmath>
#include <cassert>
#include <vector>
#include <queue>

using namespace boost;

// -------------------------------- CONSTRUCTOR/DECONSTRUCTOR --------------------------------
Astar::Astar( int beam_resolution, float max_distance, float heuristic_weight, float cost, float fos,
              float robot_radius ) {

    this->beam_resolution = beam_resolution;
    this->max_distance = max_distance;
    this->heuristic_weight = heuristic_weight;
    this->cost = cost;
    this->fos = fos;
    this->robot_radius = robot_radius;

    // set home value to initial
    home = add_vertex(graph);
    graph[home].vertex_state.setZero();
    graph[home].vertex_state = graph[home].vertex_state.array() - 1.0;
}

Astar::~Astar() {}

// ------------------------------------- GETTERS/SETTERS ------------------------------------

// provide graph
BoostGraph::Graph Astar::get_graph() {
    return graph;
}

// create vertex
Astar::Vertex Astar::create_vertex () {
    Vertex v = add_vertex(graph);
    return v;
}

// set the new goal based on the id of the AprilTag requested
void Astar::set_goal( state& goal_state ) {
    // calculate offset in x and y
    float goal_x_m = graph[goal].vertex_state[0] + max_distance * cos(graph[goal].vertex_state[2]);
    float goal_y_m = graph[goal].vertex_state[1] + max_distance * sin(graph[goal].vertex_state[2]);

    // update goal state
    goal_state = state(goal_x_m, goal_y_m, graph[goal].vertex_state[2]);

}

// get the index of a vertex on the graph
int Astar::get_vertex_index( Vertex v ) {

    // get the IndexMap property map
    index_map = get(vertex_index, graph);

    for (auto vp = vertices(graph); vp.first != vp.second; ++vp.first) {
        std::cout << "Vertex " << *vp.first << " has index " << index_map[*vp.first] << std::endl;
    }

    // get the index of the vertex
    int index = get(index_map, v);

    return index;
}

void Astar::set_vertex_properties( BoostGraph::Graph& graph, Vertex& v, state& state, float f, float g, float h, int prev) {
    graph[v].vertex_state = state;
    graph[v].from = prev;      // first path list item is the previous with -1 denoting current
    graph[v].g = 0;
    graph[v].h = euclidian(state, graph[goal].vertex_state);
    graph[v].f = graph[v].g + graph[v].h;
}

// ------------------------------------- ASTAR FUNCTIONS ------------------------------------

// determines next move for robot based on robot location, provided set of obstacles, and goal location
Astar::action Astar::astar_move(state& robot_state, std::vector<std::vector<state>> obstacles, state& goal_state ) {
    if (debug) { status_update(0, "STARTING MEASUREMENTS PROCESSING"); }

    Astar::action action;

    // goal setup
    set_goal(goal_state);
    float goal_dist = euclidian(robot_state, graph[goal].vertex_state);

    // use the beam search if distance is too far to park
    // implies that the state machine will call astar until this algorithm determines it is close enough to park
    if (goal_dist > max_distance * fos) {
        action = beam_search( robot_state, obstacles, action );
    }
    // command parking action once close enough
    else {
        action.next = INFO::AT_GOAL;
    }

    return action;

}

/*
 * beam search from target and current location
 *
 * Intention is to learn about paths that can be created by a limited set from each node.
 * The subsequent tree from each should evaluate the best options given a heruistic.
 * If the path were to hit an obstacle, the A* algorithm will trace back up the
 * obstacle to find the last forked node and pursue the one with the smallest euclidian.
 */
Astar::action Astar::beam_search(state& robot_state, std::vector<std::vector<state>> obstacles, Astar::action& action ) {
    if (debug) { status_update(0,"STARTING BEAM SEARCH"); }

    // initialize necessary function variables and objects
    std::priority_queue<std::pair<float,Vertex>> open_list;
    std::priority_queue<std::pair<float,Vertex>> closed_list;

    // setup home if not done so already
    if ((graph[home].vertex_state.array() + 1).isZero()) graph[home].vertex_state = robot_state;    // set home if initial beam search

    // initialize initial position and prepare for beam search
    Vertex initial = add_vertex(graph);
    float g = 0;
    float h = euclidian(robot_state, graph[goal].vertex_state) * heuristic_weight;
    float f = g + h;
    set_vertex_properties(graph, initial, robot_state, f, g, h, -1);

    open_list.push(std::make_pair(-f, initial));
    closed_list.push(std::make_pair(-f, initial));

    // flags and counter
    bool found = false;
    bool failed = false;
    int count = 0;

    while (!found) {

        // check if elements still in open list
        Vertex inspect_node;

        // exhausted opportunities for movement
        if (open_list.empty()) {
            action.next = INFO::ERROR;
            break;
        }
        // pick ideal node per heuristic and cost calculation
        else {
            inspect_node = open_list.top().second;
            open_list.pop();
        }

        // add to closed list
        closed_list.push(std::make_pair(-graph[inspect_node].f, inspect_node));

        // get information from vertex
        state node_state = graph[inspect_node].vertex_state;
        f = graph[inspect_node].f;
        g = graph[inspect_node].g;
        h = graph[inspect_node].h;
        count += 1;

        if (debug) { status_update(1, "INSPECTING NODE %i AT %f, %f, %f WITH f%f, g%f, h%f",
                                 inspect_node,
                                 graph[inspect_node].vertex_state[0],
                                 graph[inspect_node].vertex_state[1],
                                 graph[inspect_node].vertex_state[2],
                                 f, g, h); }



        // determine range of steering angles to pursue
        Eigen::VectorXf angles = Eigen::VectorXf::LinSpaced(beam_resolution * 2 + 1,-beam_range, beam_range);

        // TODO (P3): remove temp print
        if (verbose) {
            std::cout << "ANGLES: ";
            for (int i = 0; i < angles.size(); i++) {
                std::cout << angles(i) << " ";
            }
            std::cout << std::endl;
        }

        for ( float a : angles ) {

            // project potential next state
            float s2 = node_state[2] + a;
            float d2 = max_distance;
            float x2 = d2 * cos(s2) + node_state[0];
            float y2 = d2 * sin(s2) + node_state[1];
            state next_state = {x2, y2, s2};

            if (debug) { status_update(2, "INSPECTING MOVE %f m, %f rad to %f, %f, %f",
                                     d2, a,
                                     x2, y2, s2); }

            // check potential node
            if (!node_test(next_state, obstacles, robot_radius, fos)) continue;

            // check potential edge
            if (!edge_test(node_state, next_state, obstacles, robot_radius, fos)) continue;

            // distance to goal
            float euc_dist_m = euclidian(next_state, graph[goal].vertex_state) * heuristic_weight;

            // check if in closed list setup
            bool state_in_closed_list = false;
            std::vector<Vertex> vertices = boostGraph.get_vertices();
            for (Vertex v : vertices) {
                if (approximatelyEqual(graph[v].vertex_state[0], next_state[0]) &&
                    approximatelyEqual(graph[v].vertex_state[1], next_state[1]) &&
                    approximatelyEqual(graph[v].vertex_state[2], next_state[2])) {

                    state_in_closed_list = true;
                }
            }

            // check if state already present
            if (!state_in_closed_list) {

                // vertex creation
                float g2 = g + cost;
                float h2 = euc_dist_m;
                float f2 = g2 + h2;

                if (debug) { status_update(3,"MOVE %f m, %f rad to %f, %f, %f ADDED WITH f%f, g%f, h%f",
                                         d2, a,
                                         x2, y2, s2,
                                         f2, g2, h2); }

                Vertex next_node = add_vertex(graph);
                set_vertex_properties(graph, next_node, next_state, f2, g2, h2, inspect_node);

                // update lists
                open_list.push(std::make_pair(-f2,next_node));

                // check if within range of goal
                if ( euc_dist_m < max_distance * fos ) {
                    if (debug) {
                        status_update(
                                4,
                                "REACHED %f, %f W/IN RANGE OF TARGET %f, %f",
                                node_state[0],
                                node_state[1],
                                graph[goal].vertex_state[0],
                                graph[goal].vertex_state[1]);
                    }
                    found = true;

                    // determine next action
                    int action_idx;
                    int prev = graph[next_node].from;
                    while (prev != -1) {
                        action.path.push_back(prev);
                        action_idx = prev;
                        prev = graph[prev].from;
                    }
                    action.distance = euclidian(graph[initial].vertex_state, graph[action_idx].vertex_state);
                    action.steering = graph[initial].vertex_state[2] -
                                      graph[action_idx].vertex_state[2];   // TODO H2: EVALUATE ME
                    action.goal = graph[goal].vertex_state;
                    action.next = INFO::NA;     // NA to continue along path

                    if (debug) {
                        std::printf("%s UPCOMING PATH:\n", timestamp().c_str());
                        for (const auto &node: action.path) {
                            std::printf("\tNODE: %i", node);
                            std::printf("\tLOC: %f, %f, %f\n", graph[node].vertex_state[0],
                                        graph[node].vertex_state[1],
                                        graph[node].vertex_state[2]);
                        }
                    }

                    return action;
                }
            }
        }
    }
    graph.clear();
    return action;
}


/*
int main (int argc, char** argv) {
    Astar* robot_nav;
    state robot_state;

    robot_nav = new Astar( 2, 0.010, 1,
                           1, 1, 0.015);

    robot_state = state(0.0f, 0.0f, 0.0f);

    // straight line to goal
    state goal_state = state(.2, 0, 0);
    robot_nav->set_goal(goal_state);

    // run beam search
    std::vector<std::vector<state>> obstacles =
            {
                {state( 0.025, 0.025, 0),
                 state( 0.025, 0.050, 0),
                 state(-0.025, 0.050, 0),
                 state(-0.025, 0.025, 0)}
            };
    Astar::action next_action = robot_nav->astar_move(robot_state, obstacles, goal_state);

}*/
