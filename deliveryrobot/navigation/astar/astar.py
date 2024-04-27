"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
[Usage Description]

Classes:
[Class descriptions]

Functions:
[Provide a list of functions in the module/package with a brief description of each]

Attributes:
[Provide a list of attributes in the module/package with a brief description of each]

Dependencies:
[Provide a list of external dependencies required by the module/package]

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""

from utilities.utilities import *
from utilities.computational_geometry import test_node, test_edge, relative_angle, euclidian

import sys
sys.path.append('/opt/homebrew/Cellar/graph-tool/2.59_2/lib/python3.12/site-packages')
from graph_tool.all import *

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
        self.next = next                    # communication for 
        self.distance_m = distance_m
        self.steering_rad = steering_rad
        self.path = path
        self.goal_state = goal_state

class Astar( Component ):

    def __init__( self, beam_resolution, beam_range, max_distance, heuristic_weight, 
                 cost, fos, robot_radius_m ):
        
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

        # graph preparation
        self.G = Graph(directed=False)

        # create the vertex property maps
        self.state = self.G.new_vertex_property("vector<float>")  # NumPy array
        self.f = self.G.new_vertex_property("float")  # Float 1
        self.g = self.G.new_vertex_property("float")  # Float 2
        self.h = self.G.new_vertex_property("float")  # Float 3
        self.prev = self.G.new_vertex_property("int")  # Integer

        # create home and goal states
        self.home = self.create_vertex()
        self.goal = self.create_vertex()
        self.state[self.home] = np.zeros(3, dtype=np.float64) - 1.
        self.state[self.goal] = np.zeros(3, dtype=np.float64) - 1.


    def create_vertex( self ):
        """
        Returns:
            graph_tool.vertex: vertex object for graph

        """
        node = self.G.add_vertex()
        return node
    
    def set_goal ( self, goal_state: State ):
        """
        Sets the goal basd on new information. Goal is offset in front of
        the goal by the offset.

        Args:
            goal_state (State): state of goal

        """
        # calculate offset in x and y
        goal_x_m = goal_state[0] + self.offset * np.cos(goal_state[2])
        goal_y_m = goal_state[1] + self.offset * np.sin(goal_state[2])

        # update goal
        self.state[self.goal] = np.array([goal_x_m, goal_y_m, goal_state[2]])

    def get_vertex_properties( self, v ):
        """
        gets the vertex properties

        Returns:
            v (graph-tool.vertex): vertex object to be updated
            state (State): state information
            f (float): total cost
            g (float): action cost
            h (float): heuristic cost
            prev (int): idx of previous vertex

        """

        # updates
        state = self.state[v]
        f = self.f[v]
        g = self.g[v]
        h = self.h[v]
        prev = self.prev[v]

        return np.array(state), f, g, h, prev
    
    def set_vertex_properties( self, v, state, f, g, h, prev ):
        """
        Sets the vertex properties, updating all maps

        Args:
            v (graph-tool.vertex): vertex object to be updated
            state (State): state information
            f (float): total cost
            g (float): action cost
            h (float): heuristic cost
            prev (int): idx of previous vertex

        """

        # updates
        self.state[v] = state
        self.f[v] = f
        self.g[v] = g
        self.h[v] = euclidian( state[:2], np.array(self.state[self.goal])[:2])
        self.prev[v] = prev

    def astar_move( self, search, robot_state, goal_state, obstacles ):
        """
        determines next move for robot based on robot location, 
        provided set of obstacles, and goal location

        Args:
            search (func): search function for implementation
            robot_state (State): current state of the robot
            obstacles (StateDict): identified obstacles and states
            goal_state (State): estimated state of goal

        """

        if debug: print_status(0, "STARTING MEASUREMENTS PROCESSING")

        # initialize the output action
        action = Action()

        # goal setup
        self.set_goal(goal_state)
        goal_dist = euclidian(self.state[self.home][:2], self.state[self.goal][:2])

        # home setup
        self.state[self.home] = robot_state

        # check if close enough to park
        if goal_dist <= self.max_offset:
            action.next = INFO.AT_GOAL
        # use search method and astar to find paths
        else:
            action = search( action, obstacles )

        return action

    def beam_search( self, action, obstacles):
        """
        Intention is to learn about paths that can be created by a limited set from each node.
        The subsequent tree from each should evaluate the best options given a heruistic.
        If the path were to hit an obstacle, the A* algorithm will trace back up the
        obstacle to find the last forked node and pursue the one with the smallest euclidian.

        Args:
            action (Action): blank object with information about next steps
            obstacles (StateDict): identified obstacles and states

        Returns:
            Action: object with information about next steps

        """
        
        if debug: print_status(0, "STARTING BEAM SEARCH")

        # initialize necessary function variables and objects
        open_list = PriorityQueue()
        closed_list = PriorityQueue()

        # initialize start vertex
        start = self.create_vertex()
        g = 0
        h = euclidian(self.state[self.home][:2], self.state[self.goal][:2]) * self.heuristic_weight
        f = g + h

        self.set_vertex_properties(start, self.state[self.home], f, g, h, -1)
        open_list.put((f, start))
        closed_list.put((f, start))

        # flags and counter
        found = False
        failed = False
        count = 0

        while not found or count > 3:

            # create 
            inspect_node = self.create_vertex()

            print("open list size", open_list.qsize())

            # exhausted opportunities for movement
            if open_list.empty():
                action.next = INFO.NOT_AT_GOAL
                break
            # pick ideal node per heuristic and cost calculation
            else:
                _, inspect_node = open_list.get()

            print("open list size", open_list.qsize())

            # add to closed list
            closed_list.put((-self.f[inspect_node], inspect_node))

            # get information from vertex
            vertex_state, f, g, h, prev = self.get_vertex_properties(inspect_node)
            count += 1

            if debug: print_status(1, f"INSPECTING NODE {inspect_node} AT {vertex_state} WITH f: {f}, g {g}, h {h}")
                
            # determine range of steering angles to pursue
            angles = np.linspace(-self.beam_range, self.beam_range, self.beam_resolution * 2 + 1)

            print(f"WITH GOAL {self.state[self.goal]} AT {norm(np.array(self.state[inspect_node]) - np.array(self.state[self.goal]))} AWAY")
            input(self.get_vertex_properties(inspect_node))

            for i, a in enumerate(angles):

                # project potential next state
                s2 = vertex_state[2] + a
                d2 = self.max_distance
                x2 = d2 * np.cos(s2) + vertex_state[0]
                y2 = d2 * np.sin(s2) + vertex_state[1]
                next_state = [x2, y2, s2]

                if debug: print_status(2, f"INSPECTING MOVE {d2} m, {a} rad to {next_state}")

                # check potential node
                if not test_node(vertex_state, obstacles, self.robot_radius_m, self.fos): continue
                if not test_edge(vertex_state, next_state, obstacles, self.robot_radius_m, self.fos): continue

                # distance to goal
                euc_dist_m = euclidian(next_state[:2], self.state[self.goal][:2])
                    
                g2 = g + self.cost
                h2 = euc_dist_m * self.heuristic_weight
                f2 = g2 + h2

                if debug: print_status(3, f"MOVE {d2} m, {a} rad to {next_state} ADDED WITH f: {f2}, g: {g2}, h: {h2}")

                # create vertex instance
                next_vertex = self.create_vertex()
                self.set_vertex_properties(next_vertex, next_state, f2, g2, h2, inspect_node)

                # update lists
                open_list.put((f2, next_vertex))
                
                # check if within range of goal
                if euc_dist_m < self.max_offset:
                    if debug: print_status(4, f"REACHED {next_state} W/IN RANGE OF TARGET {self.state[self.goal]}")

                    found = True

                    # determine next action
                    action_idx = -1
                    prev = self.prev[next_vertex]
                    while prev != -1:
                        action.path.insert(0,prev)
                        action_idx = prev
                        prev = self.prev[prev]

                    action.distance = euclidian(self.state[action_idx][:2], np.array([self.state[start]])[:2])
                    action.steering = relative_angle(self.state[start], self.state[action_idx])
                    action.goal = self.state[self.goal]
                    self.next = INFO.NA

                    if debug:
                        print_status(1, f"UPCOMING PATH:\n")
                        for node in action.path:
                            print(f"\tNODE: {node}\n\
                                    \tLOC: {self.state[node]}\n")
                
                if verbose: input(f"PAUSE AT {count}")

        # clear graph for next move
        self.G.clear()

        return action
    
    # euclidian heuristic based on
    # http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
    def heuristic(self, location, goal):

        r, c, _ = location
        x, y, _ = goal

        dx = abs(x - r)
        dy = abs(y - c)

        heuristic = self.heuristic_weight * np.sqrt(dx**2 + dy**2)

        return heuristic