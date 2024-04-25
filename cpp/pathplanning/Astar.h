//
// Created by Andrew Quintana on 1/16/24.
//
// Header and implementation files use boost quick tour page:
// https://www.boost.org/doc/libs/1_75_0/libs/graph/doc/quick_tour.html
//

#ifndef JETBOTPARKING_ASTAR_H
#define JETBOTPARKING_ASTAR_H

#include "BoostGraph.h"
#include "Utilities.h"
#include "ComputationalGeometry.h"

#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include <map>
#include "Eigen/Dense"

class Astar : public Component {
protected:
    // ---------------------------------- STRUCTURES AND TYPES ----------------------------------

    BoostGraph boostGraph;
    BoostGraph::Graph& graph = boostGraph.g;

    using Vertex = BoostGraph::Vertex;
    using Edge = BoostGraph::Edge;
    using IndexMap = BoostGraph::IndexMap;
    Vertex home;
    Vertex goal;
    IndexMap index_map;

public:

    // -------------------------------- CONSTRUCTOR/DECONSTRUCTOR --------------------------------
    Astar(
            int beam_resolution,
            float max_distance,
            float heuristic_weight,
            float cost,
            float fos,
            float robot_radius);
    ~Astar();

    static struct action {
        INFO next = INFO::NA;   // provide communication to state machine
        float distance;                                 // distance for move
        float steering;                                 // steering or rotation for move
        std::list<int> path;                            // path provided for
        state goal;                                     // goal location to be compared to that
                                                        // synthesized by slam
        // this will help determine if a rerun of the astar algorithm is necessary
    };


    // ------------------------------------- GETTERS/SETTERS ------------------------------------
    void set_vertex_properties(  BoostGraph::Graph& graph, Vertex& v, state& state, float f, float g, float h, int prev );
    int get_vertex_index( Vertex v );
    BoostGraph::Graph get_graph();

    // ---------------------------------- GRAPH ----------------------------------
    Vertex create_vertex();
    void set_goal( state& goal_state );

    // ------------------------------------- ASTAR FUNCTIONS ------------------------------------
    action astar_move(state& robot_state, std::vector<std::vector<state>> obstacles, state& goal_state );

private:

    // --------------------------------------- PARAMETERS ---------------------------------------
    int beam_resolution;
    float beam_range = M_PI_4;
    float max_distance;
    float robot_radius;
    float heuristic_weight;
    float cost;
    float fos;

    // ------------------------------------- ASTAR FUNCTIONS -------------------------------------

    action beam_search(state& robot_state, std::vector<std::vector<state>> obstacles, Astar::action& action );


};


#endif //JETBOTPARKING_ASTAR_H
