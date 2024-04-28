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

import CGAL.CGAL_Kernel
from utilities.utilities import *

import numpy as np
from numpy.linalg import *
import CGAL
from CGAL.CGAL_Kernel import Vector_2, Point_2, Polygon_2, Line_2, squared_distance

logging = True
debug = True
verbose = False

def euclidian( state1, state2 ):

    if debug: print(state1, state2)
    x1_m, y1_m, _ = state1
    x2_m, y2_m, _ = state2

    dx = abs(x1_m - x2_m)
    dy = abs(y1_m - y2_m)

    heuristic = np.sqrt(dx**2 + dy**2)

    return heuristic

def relative_angle( u, v ):
    """
    calculates the angle (in radians) between two n-dimensional vectors.
    
    Args:
        u (np.ndarray): The first vector.
        v (np.ndarray): The second vector.
    
    Returns:
        float: The angle (in radians) between the two vectors, or None if one of the vectors is the zero vector.
    """
    
    if len(u) == 2:
        # normalize to u
        vector = v - u

        # calculate angle
        angle_rad = np.arctan2(vector[1],vector[0])

    else:

        if norm(u) == 0:
            angle_rad =  np.arctan2(v[1],v[0])
        elif norm(v) == 0:
            angle_rad = np.arctan2(u[1],u[0]) + np.pi
        else:
            # Calculate the unit vectors
            unit_u = u / np.linalg.norm(u)
            unit_v = v / np.linalg.norm(v)

            # Calculate the angle between the unit vectors
            angle_rad = np.arccos(np.clip(np.dot(unit_u, unit_v), -1, 1))

            # Determine the sign of the angle
            cross_product = np.cross(unit_u, unit_v)
            angle_sign = np.sign(cross_product)
            angle_rad *= angle_sign
    
    # normalize the angle
    angle_rad = normalize_angle(angle_rad)

    return angle_rad

def normalize_angle( angle_rad ):
    """
    normalize the angle between [-pi,pi)
    
    Args:
        angle (float): angle to normalize
    
    Returns:
        float: angle between [-pi,pi)
    """
    angle_rad = (angle_rad + np.pi) % (2 * np.pi)
    if angle_rad < 0:
        angle_rad += 2 * np.pi
    return angle_rad - np.pi

def collinear( a, b, c ):
    """
    Determines whether or not the three position vectors are collinear

    Args:
        a (np.ndarray[float]): point vector
        b (np.ndarray[float]): point vector
        c (np.ndarray[float]): point vector

    Returns:
        output (bool)
    """

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)

    return CGAL.CGAL_Kernel.collinear(Point_2(a[0],a[1]), Point_2(b[0],b[1]), Point_2(c[0],c[1]))

def distance_point_to_segment( point, a, b) -> float:
    """
    Calculate the distance between a point and a line segment.
    Implementation from Georgia Tech CS7632

    Args:
        point (np.ndarray): point from which to measure
        segment_point_a (np.ndarray): first point on line
        segment_point_b (np.ndarray): second point on line

    Returns:
        float: The distance between the point and the line segment.
    """
    # convert to Point_2
    l2 = norm(b - a) ** 2
    line_start_to_point_m = point - a

    # return minimum distance between line segment and point
    if l2 == 0: return norm(line_start_to_point_m)

    # calculate projection
    ab = b - a
    t = max(0, min(1, np.dot(line_start_to_point_m, ab) / l2))
    projection = a + t * ab

    return norm(point - projection)

def intersects( a, b, c, d):
    """
    determine whether or not line ab intersects with line cd

    Args:
        a (np.ndarray[float]): point vector
        b (np.ndarray[float]): point vector
        c (np.ndarray[float]): point vector
        d (np.ndarray[float]): point vector

    Returns:
        output (bool)
    """

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)
    d = d.astype(np.float64)

    # define lines
    if left(a, b, c) != left(a, b, d) and left(c, d, a) != left(c, d, b):
        return True
    else:
        return False

def generate_visibility_points( robot_radius_m, fos, obstacle ):
    """
    Get the visibility point from a vertex and neighboring points
    The visibility points should be offset from the vertex so that each point is
    offset from the vertex by robot_radius * fos

    Args:
        robot_radius_m (float): radius of the robot
        fos (float): factor of safety of robot positioning
        obstacle (np.ndarray[np.ndarray[float]]): array of obstacle points

    Returns:
        np.ndarray[float]: visibility points of polygon
    """

    obstacle_povs = np.empty((len(obstacle),3),dtype=np.float64)

    # iterate through obstacle points, selecting points on either side
    for b in range(len(obstacle)):
        a = len(obstacle) - 1 if b == 0 else b - 1      # one to CW dir
        c = 0 if b == len(obstacle) - 1 else b + 1      # one to CCW dir

        viz = visibility_point(robot_radius_m, fos, obstacle[a], obstacle[b], obstacle[c])
        obstacle_povs[b] = viz

    return obstacle_povs

def left( a, b, c ):
    """
    function to check if point c is on the left of line segment ab

    Args:
        a (np.ndarray[float]): point vector
        b (np.ndarray[float]): point vector
        c (np.ndarray[float]): point vector

    Returns:
        output (bool)
    """

    return left_math(a,b,c) > 0

def left_on( a, b, c ):
    """
    function to check if point c is on the left of line segment ab or on it

    Args:
        a (np.ndarray[float]): point vector
        b (np.ndarray[float]): point vector
        c (np.ndarray[float]): point vector

    Returns:
        output (bool)
    """

    return left_math(a,b,c) >= 0

def left_math( a, b, c ): return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def test_edge(line_start, line_end, obstacles, agent_radius_m, fos):
    """
    Determine if a line segment is valid, i.e., not too close to obstacles and not intersecting any obstacles.

    Args:
        line_start (numpy.ndarray): The start point of the line segment as a 1D NumPy array with shape (2,).
        line_end (numpy.ndarray): The end point of the line segment as a 1D NumPy array with shape (2,).
        obstacles (list[list[numpy.ndarray]]): A list of obstacle points, where each obstacle is represented as a list of 1D NumPy arrays with shape (2,).
        agent_radius_m (float): The radius of the agent in meters.
        fos (float): The factor of safety for the agent radius.

    Returns:
        bool: True if the line segment is valid, False otherwise.
    """

    for obstacle_points in obstacles.values():
        for i in range(len(obstacle_points)):
            j = (i + 1) % len(obstacle_points)
            if verbose:
                print(f"OBSTACLE VERTEX TO EDGE TEST:\tOB VERTEX: \
                      ({obstacle_points[i][0]:.2f}, {obstacle_points[i][1]:.2f})\t \
                      EDGE ({line_start[0]:.2f}, {line_start[1]:.2f})-({line_end[0]:.2f}, {line_end[1]:.2f})")

            if distance_point_to_segment(obstacle_points[i], line_start, line_end) < agent_radius_m * fos:
                if debug:
                    print(f"OBSTACLE VERTEX TOO CLOSE TO EDGE:\tOB VERTEX: \
                          ({obstacle_points[i][0]:.2f}, {obstacle_points[i][1]:.2f})\t \
                            EDGE ({line_start[0]:.2f}, {line_start[1]:.2f})-({line_end[0]:.2f}, {line_end[1]:.2f})")
                return False

            if verbose:
                print(f"NODE EDGE INTERSECTION TEST:\tNODE EDGE: \
                      ({line_start[0]:.2f}, {line_start[1]:.2f})-({line_end[0]:.2f}, {line_end[1]:.2f})\t \
                        OBS EDGE: ({obstacle_points[i][0]:.2f}, \
                                   {obstacle_points[i][1]:.2f})-({obstacle_points[j][0]:.2f}, \
                                   {obstacle_points[j][1]:.2f})")

            if intersects(line_start, line_end, obstacle_points[i], obstacle_points[j]):
                if debug:
                    print(f"NODE EDGE INTERSECTS WITH OBSTACLE EDGE:\tNODE EDGE: \
                          ({line_start[0]:.2f}, {line_start[1]:.2f})-({line_end[0]:.2f}, {line_end[1]:.2f})\t \
                            OBS EDGE: ({obstacle_points[i][0]:.2f}, \
                                       {obstacle_points[i][1]:.2f})-({obstacle_points[j][0]:.2f}, \
                                       {obstacle_points[j][1]:.2f})")
                return False

    return True

def test_node(node, obstacles, agent_radius_m, fos):
    """
    Determine if a node is within the environment and not too close to the boundary or obstacles.

    Args:
        node (numpy.ndarray): The node as a 1D NumPy array with shape (2,).
        obstacles (list[list[numpy.ndarray]]): A list of obstacle points, where each obstacle is represented as a list of 1D NumPy arrays with shape (2,).
        agent_radius_m (float): The radius of the agent in meters.
        fos (float): The factor of safety for the agent radius.

    Returns:
        bool: True if the node is valid, False otherwise.
    """
    # TODO (P3): determine if node within environment

    # TODO (P3): determine if node too close to boundary of environment

    for obstacle_points in obstacles.values():
        left_count = 0

        for i in range(len(obstacle_points)):
            j = (i + 1) % len(obstacle_points)

            if distance_point_to_segment(node, obstacle_points[i], obstacle_points[j]) < agent_radius_m * fos:
                if debug:
                    print(f"NODE TOO CLOSE TO OBSTACLE EDGE:\tNODE: ({node[0]:.2f}, {node[1]:.2f})\t \
                        EDGE ({obstacle_points[i][0]:.2f}, \
                                {obstacle_points[i][1]:.2f})-({obstacle_points[j][0]:.2f}, \
                                {obstacle_points[j][1]:.2f})")
                return False

            if left_on(node, obstacle_points[i], obstacle_points[j]):
                left_count += 1

        if verbose:
            print(f"LEFT COUNT = {left_count} of {len(obstacle_points)}")
        if left_count < len(obstacle_points):
            if debug:
                print(f"NODE INSIDE OBSTACLE:\t \
                    NODE: ({node[0]:.2f}, {node[1]:.2f})\t \
                    OBSTACLE LEFT_ON {left_count}")

    return True


def visibility_point( robot_radius_m, fos, a, b, c ):
    """
    Get the visibility point from a vertex and neighboring points
    The visibility points should be offset from the vertex so that each point is
    offset from the vertex by robot_radius * fos

    Args:
        robot_radius_m (float): radius of the robot in meters
        fos (float): factor of safety of robot positioning
        a (np.ndarray[float]): point vector representing point CW of center
        b (np.ndarray[float]): point vector at vertex
        c (np.ndarray[float]): point vector representing point CCW of center

    Returns:
        np.ndarray[float]: single visibility point
    """
    
    # normalized to vertex
    a_norm = a - b
    c_norm = c - b

    # calculate components
    psi_a = np.arctan2(a_norm[1],a_norm[0])
    psi_c = np.arctan2(c_norm[1],c_norm[0])
    vis_len_m = robot_radius_m * fos

    
    # midpoint angle
    mid = (a + c) / 2
    mid_angle = relative_angle(b, mid)
    viz_angle = mid_angle + np.pi

    # calculate visibility point coordinates
    x_v = b[0] + vis_len_m * np.cos(viz_angle)
    y_v = b[1] + vis_len_m * np.sin(viz_angle)

    return np.array([x_v, y_v, 0],dtype=np.float64)

def xprod( a, b, c):
    """
    function to compute cross product of ab and ac

    Args:
        a (np.ndarray[float]): point vector
        b (np.ndarray[float]): point vector
        c (np.ndarray[float]): point vector

    Returns:
        output (float)
    """

    # create vectors
    ab = b - a
    ac = c - a

    return np.cross(ab, ac)