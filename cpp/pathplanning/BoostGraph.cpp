//
// Created by Andrew Quintana on 2/1/24.
//

#include "BoostGraph.h"

BoostGraph::BoostGraph() {}
BoostGraph::~BoostGraph() {}

std::vector<BoostGraph::Vertex> BoostGraph::get_vertices() {
    std::vector<Vertex> vertex_vector;

    // Iterate through the vertices of the graph
    std::pair<vertex_iter, vertex_iter> vp;
    for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
        Vertex v = *vp.first;
        vertex_vector.push_back(v);
    }

    return vertex_vector;
}

std::vector<BoostGraph::Vertex> BoostGraph::get_adjacent( Vertex v ) {
    std::vector<Vertex> adjacent_vector;

    // iterate through adjacent vertices of vertex
    adj_iter ai, ai_end;
    for (boost::tie(ai, ai_end) = adjacent_vertices(v, g); ai != ai_end; ++ai) {
        adjacent_vector.push_back(*ai);
    }

    return adjacent_vector;
}

std::vector<BoostGraph::Edge> BoostGraph::get_edges() {
    std::vector<Edge> edge_vector;

    // Iterate through the edges of the graph
    std::pair<edge_iter, edge_iter> ei;
    for (ei = edges(g); ei.first != ei.second; ++ei.first) {
        edge_vector.push_back(*ei.first);
    }
    return edge_vector;
}