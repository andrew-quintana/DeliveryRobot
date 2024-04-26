"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Visualization method for SLAM, A* and other AI methods

Classes:
Mapping: class for generating a mapping of environment

Functions:
[Provide a list of functions in the module/package with a brief description of each]

Attributes:
[Provide a list of attributes in the module/package with a brief description of each]

Dependencies:
[Provide a list of external dependencies required by the module/package]

License:
[Include the full text of the license you have chosen for your code]

Examples:
 - plot_radar: provides high level functionality of class with provided arguments

"""

from utilities.utilities import *

import cv2
import numpy as np
import math

# dictionary for the plotters inputs
color_dict = {
    "robot" : (0,0,255),
    "apriltag" : (255,0,0),
    "obstacle" : (0,0,0)
}
shape_dict = {
    "robot" :       np.array([[ 0, -3],
                              [ 2,  2],
                              [-2,  2]]),
    "apriltag" :    np.array([[ 0, -3],
                              [ 2,  2],
                              [-2,  2]]),
    "obstacle" :    np.array([[ 0, -3],
                              [ 2,  2],
                              [-2,  2]])
}

class Map( Component ):
    """
    Mapping class for depciting the robot's position relative to landmarks in the environment.
    """

    def __init__( self, window_name, live ):
        """
        Constructor of the map

        Args:
            window_name (str): providing the name of the map window
            live (bool): should only the most recent map be saved?
        """

        self.window_name = window_name
        self.live = live

        self.n = 0
        self.path = ""

    def coord_text( self, color, id, c_x, c_y, psi):
        """
        Add text at specific coordinate based on center point of map element to label

        Args:
            color (np.ndarray[int]): color array
            id (str): string representing the labeled landmark or robot
            c_x (float): center point on the x axis
            c_y (float): center point on the y axis
            psi (float): yaw rotation or on the x-y plane
            scale (int): scale of text

        """

        # get center and image information
        rows, cols = self.img.shape[:2]
        img_cx = cols/2
        img_cy = rows/2

        # define text to be displayed
        text = f"{id} ({c_x:.2f}, {c_y:.2f}, {psi:.2f})"

        # define font properties
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        thickness = 1
        scale = 1.25

        # calculate position to place on text
        offset = 10 if c_x > 10 else -325
        org = (int(img_cx + c_x + offset), int(img_cy + c_y))
        self.img = cv2.putText(self.img, text, org, font, scale, color, thickness, cv2.LINE_AA)

    
    def plot_element( self, state, type, id, scale):
        """
        Plotter of a map element

        Args:
            state (state): position vector of the element
            type (str): type of element description
            id (str): id of element
        """

        # get center and image information
        rows, cols = self.img.shape[:2]
        img_cx = cols/2
        img_cy = rows/2

        # define element map element attributes
        c_x, c_y, psi = -state[0], -state[1], state[2]
        color = color_dict[type]

        # create shape coordinates and rotate to desired position and scale
        polygon_coords = shape_dict[type]
        angle = psi
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        polygon_coords = np.dot(R, polygon_coords.T).T

        # scale and offset polygon
        polygon_coords = polygon_coords * scale + np.array([img_cx + c_x, img_cy + c_y])
        
        # draw filled polygon on image
        cv2.fillPoly(self.img, [polygon_coords.astype(np.int32)], color)

        # add text
        if (type == "robot"):
            label = "R" + id
        elif (type == "apriltag"):
            label = "A" + id
        else:
            label = "O" + id
    
        self.coord_text(color, label, c_x, c_y, psi)

    def plot_grid( self, spacing, thickness):
        """
        Plotter of grid for map readability

        Args:
            img (np.ndarray[np.uint8,np.unit8]): image array
            spacing (int): distance between each line in grid
            thickness (int): thickness of lines
        """

        # grid characteristics
        rows, cols = self.img.shape[:2]
        opacity = 10
        color_value = 255 * opacity / 100
        color = np.ones_like((3,1),dtype=np.uint8) * color_value

        # create lines in image
        for i in range(rows):
            cv2.line(self.img, (0, i * spacing), (cols, i * spacing), color, thickness)
        
        for j in range(rows):
            cv2.line(self.img, (j * spacing, 0), (j * spacing, rows), color, thickness)    


    def plot_radar( self, env):
        """
        High level plotter for running all of the processes

        Args:
            env (StateDict): list of states in environment
            save_path (str): 

        """

        if self.live and self.path != "": os.remove(self.path)
        else: self.n += 1

        # create array
        self.img = np.ones((1000,1000,3),dtype=np.uint8) * 255

        # plot grid
        self.plot_grid( 10, 1)

        # iterate through environment elements
        for key in env:

            # identify type and id
            if key == "ROBOT":
                type = "robot"
                id = "1"      # TODO update if multiple robots desired
            elif key in obstacle_ids:
                type = "obstacle"
                id = key
            else:
                type = "apriltag"
                id = key
                
            # plot landmarks
            self.plot_element(env[key], type, id, 3)

        if self.live: self.path = save(self.img, "map", filename="live_map.png")
        else: save(self.img, "map", "iter", filename=f"map_{self.n}")

# TODO Fix me
def generate_obstacle( tag_state: State, side_length_m):
    """
    determine whether or not line ab intersects with line cd

    Args:
        tag_state (State): center point of the tag
        side_length_m (float): length of virtual obstacle

    Returns:
        np.ndarray: ccw points of obstacle
    """

    # Extract x, y, and theta from the state vector
    x, y, psi = tag_state

    # Calculate the coordinates of the four corners of the square
    half_side_length = side_length_m / 2
    corners = np.array([[x - half_side_length, y - half_side_length],
                        [x + half_side_length, y - half_side_length],
                        [x + half_side_length, y + half_side_length],
                        [x - half_side_length, y + half_side_length]], dtype=np.float64)

    # Create the rotation matrix
    rotation_matrix = rotation_matrix_2d(psi)

    # Rotate the corners of the square about the state coordinate
    rotated_corners = np.dot(rotation_matrix, corners.T).T.astype(np.float64)

    return np.array(rotated_corners)
    

def translation_matrix_2d( dx, dy ):
    """
    Create a 3x3 translation matrix for 2d translations

    Args:
        tx (float): x axis translation
        ty (float): y axis translation

    Returns:
        np.ndarray: 3x3 translation matrix.
    """

    T = [[1, 0, dx],
         [0, 1, dy],
         [0, 0,  1]]
    return np.array(T, dtype=float)

def rotation_matrix_2d( angle ):
    """
    Create a 3x3 rotation matrix for 2d rotations

    Args:
        angle (float): rotation angle in radians

    Returns:
        np.ndarray: 3x3 rotation matrix
    """

    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]])
    
    # apply threshold for near-zero values
    threshold = 1e-10
    R[np.abs(R) < threshold] = 0

    return R