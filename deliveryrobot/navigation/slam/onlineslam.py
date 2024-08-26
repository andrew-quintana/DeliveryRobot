"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Use OnlineSlam algorithm for localization and mapping.

Classes:
    OnlineSlam: houses attributes and functions for algorithm
        get_map(): get map externally
        process_measurements(): process information from sensor readings
        process_movement(): process movement from sensors or estimate
        map_update(): update map based on class matrices and algorithm linear algebra

Dependencies:
    utilities.py

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""

from utilities.utilities import *

import cv2
import numpy as np
from numpy.linalg import inv
import math

class OnlineSLAM( Component ):

    def __init__( self, dim ):
        """
        Constructor of the online slam instance
        """
        super().__init__()
        
        # matrices for calculation
        self.mu = np.zeros((dim,1), dtype=np.float64)
        self.Omega = np.eye(dim, dtype=np.float64)
        self.Xi = np.zeros((dim,1), dtype=np.float64)

        # dimension
        self.dim = dim

        # environment information
        self.map = {"ROBOT": np.array([70., -220., 8.])}     # tracks all states
        self.landmarks = {"ROBOT": 0}                               # tracks indices in array

    def get_map( self ):
        """
        Returns:
            dict(str, ndarray): map of all of the states in the environment

        """

        return self.map
    
    def process_measurements( self, measurements ):
        """
        process information from scan

        Args:
            measurements (dict(str, ndarray)): measurements from the last scan

        """

        if self.debug: print("MEASUREMENTS PROCESSING")
        for key in measurements:

            # do not include robot location
            if key == "ROBOT": continue
            
            # create new landmark if first sighting
            if key not in self.map:

                # add to tracked map and landmarks list
                self.map[key] = measurements[key]

                # append rows and columns to Omega
                self.Omega = insert_rows_cols(self.Omega, self.Omega.shape[0], self.Omega.shape[1], self.dim, self.dim)
                self.Xi = insert_rows_cols(self.Xi, self.Xi.shape[0], 0, self.dim, 0)
                self.landmarks[key] = self.Omega.shape[0] - self.dim

            # integrate measurements
            # credit given to OMSCS 7638 staff
            idx = self.landmarks[key]
            for i in range(self.dim):
                self.Omega[i,i] += 1
                self.Omega[idx + i, idx + i] += 1
                self.Omega[idx + i, i] += -1
                self.Omega[i, idx + i] += -1
                self.Xi[i, 0] += -measurements[key][i]
                self.Xi[idx + i, 0] += measurements[key][i]

            if self.debug: print("Omega\n", self.Omega)
            if self.debug: print("Xi\n", self.Xi)

        return True


    def process_movement( self, translation_m, rotation_rad ):
        """
        process movement commanded

        Args:
            measurements (dict(str, ndarray)): measurements from the last scan

        """

        if self.debug: print("MOVEMENT PROCESSING")

        # determine theoretical new position
        new_robot_bearing_rad = self.map["ROBOT"][2] + rotation_rad
        new_robot_delta_x_m = translation_m * np.cos(new_robot_bearing_rad)
        new_robot_delta_y_m = translation_m * np.sin(new_robot_bearing_rad)
        estimate = np.array([new_robot_delta_x_m, new_robot_delta_y_m, new_robot_bearing_rad])

        # expand information and vector matrices by one new position
        self.Omega = insert_rows_cols(self.Omega, self.dim, self.dim, self.dim, self.dim)
        self.Xi = insert_rows_cols(self.Xi, self.dim, 0, self.dim, 0)

        # update information matrix/vector based on the robot motion
        for i in range(self.dim * 2):
            self.Omega[i, i] += 1

        for i in range(self.dim):
            self.Omega[self.dim + i, i] += -1
            self.Omega[i, self.dim + i] += -1
            self.Xi[i, 0] += -estimate[i]
            self.Xi[self.dim + i, 0] += estimate[i]

        if self.debug: print("Omega\n", self.Omega)
        if self.debug: print("Xi\n", self.Xi)

        return True

    def map_update( self, motion=True ):
        """
        update map based on processing of class matrices

        """

        if self.debug: print("MAPPING UPDATE")

        if motion:

            # determine submatrices for calculation
            newidxs = list(range(self.dim, len(self.Omega)))
            a = self.Omega[0 : self.dim, self.dim :]
            b = self.Omega[0 : self.dim, 0 : self.dim]
            c = self.Xi[0:self.dim, :]
            pOmega = self.Omega[self.dim :, self.dim :]
            pXi = self.Xi[self.dim :, :]

            # calculate Omega and Xi
            self.Omega = pOmega - np.matmul(np.matmul(a.T, inv(b)), a)
            self.Xi = pXi - np.matmul(np.matmul(a.T, inv(b)), c)
        
        # determine mu
        self.mu = np.matmul(inv(self.Omega), self.Xi)
        if self.debug: print(f"Calculated mu as:\n{self.mu}")

        # map update
        for key in self.landmarks:
            idx = self.landmarks[key]
            for i in range(self.dim):
                self.map[key][i] = self.mu[idx + i, 0]

        if self.debug: print("map\n", self.map)

        return True