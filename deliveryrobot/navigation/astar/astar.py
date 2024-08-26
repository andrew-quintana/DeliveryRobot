"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Path planning based on Astar and Beam Search algorithms.

Classes:
    Graph: local graph datatype
        add_vertex(): add vertex to graph
        add_edge(): add edge to graph
        get_vertex_properties(): get information about vertex
        clear(): clear graph
    Astar: contains algorithms for beam search and astar
        reset(): reset/delete relevant inputs and outputs for algorithm
        add_vertex(): execute vertex addition
        plot_map(): show the map based on updated mapper
        astar_move(): determine path based on current information
        beam_search(): investigate opportunities for movement based on beam or
            whisker methodology
            - actual astar implementation WITH beam search incorporated

Functions:
    hueristic(): calculate value to determine risk weight

Dependencies:
    utilities.py
    computational_geomtry.py
    mapper.py

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""

from utilities.utilities import *
from utilities.computational_geometry import test_node, test_edge, relative_angle, euclidian
from utilities.mapper import Map

import numpy as np
from numpy.linalg import norm
from queue import PriorityQueue
import copy

class Action:
    def __init__(   self, next=INFO.NA, 
                    distance_m = 0., 
                    steering_rad = 0., 
                    path = [], 
                    goal_state = np.zeros(3, dtype=np.float64) ):
        self.next = next                    # communication for state machine
        self.distance_m = distance_m
        self.steering_rad = steering_rad
        self.path = path                    # [(idx, state)]
        self.goal_state = goal_state
        
    def print_info(self):
        print("Next: ", self.next)
        print("Distance (m): ", self.distance_m)
        print("Steering (rad): ", self.steering_rad)
        print("Path: ", self.path)
        print("Goal State: ", self.goal_state)

class Vertex:
    def __init__(self, state, f, g, h, prev, index):
        self.state = np.array(state) if not isinstance(state, np.ndarray) else state
        self.f = f
        self.g = g
        self.h = h
        self.prev = prev
        self.index = index

class Graph:
    def __init__(self):
        self.vertices = []
        self.adjacency_list = {}
        self.vertex_index_map = {}

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        self.adjacency_list[vertex] = []
        self.vertex_index_map[vertex.index] = vertex

    def add_edge(self, vertex1, vertex2):
        self.adjacency_list[vertex1].append(vertex2)
        self.adjacency_list[vertex2].append(vertex1)

    def get_vertex_properties(self, vertex):
        return vertex.state, vertex.f, vertex.g, vertex.h, vertex.prev
    
    def clear(self):
        """
        Clears the contents of the Graph object.
        """
        self.vertices.clear()
        self.adjacency_list.clear()
        self.vertex_index_map.clear()

class Astar( Component ):

    def __init__( self, beam_resolution, beam_range, max_distance, heuristic_weight, 
                 cost, fos, robot_radius_m ):
        
        super().__init__()
        
        # graph preparation
        self.graph = Graph()

        self.reset()

        # object attributes
        self.beam_resolution = beam_resolution
        self.beam_range = beam_range
        self.max_distance = max_distance
        self.heuristic_weight = heuristic_weight
        self.cost = cost
        self.fos = fos
        self.robot_radius_m = robot_radius_m

        # interpreted attributes
        self.offset = robot_radius_m * fos
        self.max_offset = max_distance * fos

        # manually assigned attributes
        self.tol = 5e-2

        

    def reset( self ):
        self.graph.clear()
        try: del(self.action)
        except: pass

    def add_vertex( self, state, f, g, h, prev ):
        """
        create the vertex with properties, updating all maps

        Args:
            v (graph-tool.vertex): vertex object to be updated
            state (ndarray): state information
            f (float): total cost
            g (float): action cost
            h (float): heuristic cost
            prev (int): idx of previous vertex

        """

        vertex = Vertex(state, f, g, h, prev, len(self.graph.vertices))
        self.graph.add_vertex(vertex)
        return vertex
    
    def set_goal ( self, goal_state ):
        """
        DON"T USE ME
        Sets the goal basd on new information. Goal is offset in front of
        the goal by the offset.

        Args:
            goal_state (State): state of goal

        """
        # calculate offset in x and y
        goal_x_m = goal_state[0] + self.offset * np.cos(goal_state[2])
        goal_y_m = goal_state[1] + self.offset * np.sin(goal_state[2])

        # update goal
        #self.state[self.goal] = np.array([goal_x_m, goal_y_m, goal_state[2]])
        self.goal_state = np.array([goal_x_m, goal_y_m, goal_state[2]])

    def plot_map(self):
        """
        Plot the map based on the Action object.
        """
        map = Map("Robot_Map", live=True)

        # Create a dictionary of the environment states
        env = {
            "ROBOT": self.robot_state,
            "GOAL": self.goal_state
        }

        # Add the path points to the environment
        for idx, state in self.action.path:
            env[f"PATH_{idx}"] = state

        # Plot the map
        map.plot_radar(env)

    def astar_move( self, robot_state, obstacles, goal_state = [] ):
        """
        determines next move for robot based on robot location, 
        provided set of obstacles, and goal location

        Args:
            search (func): search function for implementation
            robot_state (State): current state of the robot
            obstacles (StateDict): identified obstacles and states
            goal_state (State): estimated state of goal

        """

        if self.debug: print_status(0, "STARTING MEASUREMENTS PROCESSING")


        # home setup
        self.robot_state = robot_state

        # goal setup
        if goal_state != []:
            self.set_goal(goal_state)
        goal_dist = euclidian(self.robot_state, self.goal_state)
        
        # initialize the output action
        self.action = Action()
        self.action.goal_state = self.goal_state

        # check if close enough to park
        if goal_dist <= self.max_offset:
            self.action.next = INFO.AT_GOAL
        # use search method and astar to find paths
        else:
            self.beam_search( obstacles )

        return self.action

    def beam_search( self, obstacles ):
        """
        Intention is to learn about paths that can be created by a limited set from each node.
        The subsequent tree from each should evaluate the best options given a heruistic.
        If the path were to hit an obstacle, the A* algorithm will trace back up the
        obstacle to find the last forked node and pursue the one with the smallest euclidian.

        Args:
            obstacles (StateDict): identified obstacles and states

        Returns:
            Action: object with information about next steps

        """
        
        if self.debug: print_status(0, "STARTING BEAM SEARCH")

        # initialize necessary function variables and objects
        open_list = PriorityQueue()
        closed_list = PriorityQueue()

        # initialize start vertex
        g = 0
        h = euclidian(self.robot_state, self.goal_state) * self.heuristic_weight
        f = g + h

        start = self.add_vertex( self.robot_state, f, g, h, None)
        open_list.put((f, start.index))
        closed_list.put((f, start.index))

        # flags and counter
        found = False
        count = 0

        while not found:

            # exhausted opportunities for movement
            if open_list.empty():
                self.action.next = INFO.FAILED
                break
            # pick ideal node per heuristic and cost calculation
            else:
                _, inspect_node_idx = open_list.get()

            # add to closed list
            inspect_node = self.graph.vertex_index_map[inspect_node_idx]
            closed_list.put((inspect_node.f, inspect_node.index))

            # get information from vertex
            vertex_state, f, g, h, prev = self.graph.get_vertex_properties(inspect_node)
            count += 1

            if self.debug: print_status(1, f"INSPECTING NODE {inspect_node.index} AT {vertex_state} WITH f: {f}, g {g}, h {h}")
                
            # determine range of steering angles to pursue
            angles = np.linspace(-self.beam_range, self.beam_range, self.beam_resolution * 2 + 1)

            if self.debug: print(f"WITH GOAL {self.goal_state} AT {euclidian(inspect_node.state, self.goal_state)} AWAY")
            if self.verbose: input(self.graph.get_vertex_properties(inspect_node))

            for i, a in enumerate(angles):

                # project potential next state
                s2 = inspect_node.state[2] + a
                d2 = self.max_distance
                x2 = d2 * np.cos(s2) + inspect_node.state[0]
                y2 = d2 * np.sin(s2) + inspect_node.state[1]
                next_state = np.array([x2, y2, s2])

                if self.debug: print_status(2, f"INSPECTING MOVE {d2} m, {a} rad to {next_state}")

                # check potential node
                if not test_node(next_state, obstacles, self.robot_radius_m, self.fos): continue
                if not test_edge(vertex_state, next_state, obstacles, self.robot_radius_m, self.fos): continue

                # distance to goal
                euc_dist_m = euclidian(next_state, self.goal_state)
                    
                g2 = g + self.cost
                h2 = euc_dist_m * self.heuristic_weight
                f2 = g2 + h2

                # create vertex instance
                next_vertex = self.add_vertex(next_state, f2, g2, h2, inspect_node)

                if self.debug: print_status(3, f"MOVE {d2} m, {a} rad to {next_state} ADDED WITH f: {f2}, g: {g2}, h: {h2}, IDX: {next_vertex.index}")

                # update lists
                open_list.put((f2, next_vertex.index))
                
                # check if within range of goal
                if self.verbose: print(f"{euc_dist_m < self.max_offset} DISTANCE {euc_dist_m} < {self.max_offset}")
                if euc_dist_m < self.max_offset:
                    if self.debug: print_status(4, f"REACHED {next_state} W/IN RANGE OF TARGET {self.goal_state} with idx {next_vertex.index}")

                    found = True

                    # determine next action
                    self.action.path = []
                    action_idx = -1
                    prev_vertex = self.graph.vertex_index_map[next_vertex.index]
                    while prev_vertex != None:
                        self.action.path.insert(0, (prev_vertex.index, prev_vertex.state))
                        action_idx = prev_vertex.index
                        prev_vertex = self.graph.vertex_index_map[prev_vertex.index].prev
                    
                    action_vertex = self.graph.vertex_index_map[self.action.path[1][0]]
                    self.action.distance_m = euclidian(action_vertex.state, self.robot_state)
                    self.action.steering_rad = relative_angle(self.robot_state, action_vertex.state)
                    self.action.next = INFO.NA
                    
                    
                    if self.logging:
                        dist_sum_x = 0
                        dist_sum_y = 0
                        turn_sum = 0
                        prev_angle = 0
                        print_status(1, f"UPCOMING PATH:\n")
                        for node, state in self.action.path:
                            print(f"\tNODE: {node}\n\
                                    \tLOC: {self.graph.vertex_index_map[node].state}\n")

                    # save a map image
                    self.plot_map()
                    break
    
    # euclidian heuristic based on
    # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
    def heuristic(self, location):

        euc_dist = euclidian(location, self.goal_state)
        home_to_goal = euclidian(self.robot_state, self.goal_state)

        return home_to_goal - euc_dist
    
    