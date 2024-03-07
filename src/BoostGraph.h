//
// Created by Andrew Quintana on 2/1/24.
//

#ifndef JETBOTPARKING_BOOSTGRAPH_H
#define JETBOTPARKING_BOOSTGRAPH_H

#include "Utilities.h"
#include <iostream>                  // for std::cout
#include <utility>                   // for std::pair
#include <algorithm>                 // for std::for_each
#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include <map>
#include "Eigen/Dense"

using namespace boost;

class BoostGraph {
private:
    // define the properties for the vertices
    struct VertexProperties {
        state vertex_state;
        float f;
        float g;
        float h;
        int from;
    };


public:

    BoostGraph();
    ~BoostGraph();

    // define the properties for the edges
    struct EdgeProperties {
        double weight;
    };

    // define the adjacency list type with the desired properties
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, VertexProperties, EdgeProperties> Graph;
    Graph g;

    // define the Vertex type
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits<Graph>::edge_descriptor Edge;



    // define the property map for vertex indices
    typedef property_map<Graph, vertex_index_t>::type IndexMap;

    // define the iterators for looking through indices
    typedef graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef graph_traits<Graph>::edge_iterator edge_iter;
    typedef graph_traits<Graph>::adjacency_iterator adj_iter;


    std::vector<Vertex> get_vertices();

    std::vector<Vertex> get_adjacent(Vertex v);

    std::vector<Edge> get_edges();


};

#endif //JETBOTPARKING_BOOSTGRAPH_H
