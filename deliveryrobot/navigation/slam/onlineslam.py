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
from utilities.computational_geometry import *
from navigation.filters.kalman_filter import *

import cv2
import numpy as np
from numpy.linalg import inv
import math

class OnlineSLAM( Component ):

    def __init__( self, dim, meas_weight=1., move_weight=1. ):
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
        
        # weights
        self.psi_factor = 2.0
        self.meas_weight = meas_weight
        self.move_weight = move_weight
        self.meas_weight_psi = meas_weight #* self.psi_factor  # Higher weight for Psi measurements
        self.move_weight_psi = move_weight #* 1/self.psi_factor  # Lower weight for Psi motion
        
        # environment information
        self.map = {"ROBOT": np.array([0., 0., 0.])}        # tracks all states
        self.landmarks = {"ROBOT": 0}                       # tracks indices in array
        
        # world frame map information
        self.world_frame_idx = -1
        self.world_frame_map = {"ROBOT": np.array([0., 0., 0.])}

    def get_map( self ):
        """
        Returns:
            dict(str, ndarray): map of all of the states in the environment
        """

        return self.map if self.world_frame_idx == -1 else self.world_frame_map 
    
    def set_world_frame( self, world_frame_idx: int ):
        """
        
        Args:
            world_frame (int): index of the world frame for mapping wrt a specific landmark
        """

        self.world_frame_idx = world_frame_idx + 1
        print(f"World frame set to {self.world_frame_idx}")
        
    def initialize_filters( self, estimate_dict ):
        
        # initialize psi kalman filter
        self.psi_kf = KalmanFilter(process_var=0.001, measurement_var=0.01, initial_estimate=estimate_dict["psi"])

        
    def update_weights( self, meas_weight: float=1., move_weight: float=1. ):
        
        self.meas_weight = meas_weight
        self.move_weight = move_weight
        self.meas_weight_psi = meas_weight #* self.psi_factor  # Higher weight for Psi measurements
        self.move_weight_psi = move_weight #* 1/self.psi_factor  # Lower weight for Psi motion
    
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
                self.world_frame_map[key] = np.zeros_like(self.map[key], dtype=np.float64)

                # append rows and columns to Omega
                self.Omega = insert_rows_cols(self.Omega, self.Omega.shape[0], self.Omega.shape[1], self.dim, self.dim)
                self.Xi = insert_rows_cols(self.Xi, self.Xi.shape[0], 0, self.dim, 0)
                self.landmarks[key] = self.Omega.shape[0] - self.dim

            # integrate measurements
            # credit given to OMSCS 7638 staff
            idx = self.landmarks[key]
            for i in range(self.dim):
                if i == 2:  # Psi (orientation)
                    self.Omega[i, i] += self.meas_weight_psi  # Apply higher weight for Psi measurements
                    self.Omega[idx + i, idx + i] += self.meas_weight_psi
                    self.Omega[idx + i, i] += -self.meas_weight_psi
                    self.Omega[i, idx + i] += -self.meas_weight_psi
                    self.Xi[i, 0] += -measurements[key][i] * self.meas_weight_psi
                    self.Xi[idx + i, 0] += measurements[key][i] * self.meas_weight_psi
                else:
                    # X and Y measurements
                    self.Omega[i, i] += self.meas_weight
                    self.Omega[idx + i, idx + i] += self.meas_weight
                    self.Omega[idx + i, i] += -self.meas_weight
                    self.Omega[i, idx + i] += -self.meas_weight
                    self.Xi[i, 0] += -measurements[key][i]
                    self.Xi[idx + i, 0] += measurements[key][i]

            if self.debug: print("Omega\n", self.Omega)
            if self.debug: print("Xi\n", self.Xi)

        return True


    def process_movement( self, translation_m, rotation_rad, meas=True ):
        """
        process movement commanded

        Args:
            measurements (dict(str, ndarray)): measurements from the last scan

        """

        if self.debug: print("MOVEMENT PROCESSING")
            
        #if not meas: self.update_weights(move_weight=self.move_weight * 1.1)

        # determine theoretical new position
        if self.debug: print(f"PSI ESTIMATE CALCULATION:\n{self.map['ROBOT'][2]} + {rotation_rad} = {self.map['ROBOT'][2] + rotation_rad}")
        new_robot_bearing_rad = self.map["ROBOT"][2] + rotation_rad
        new_robot_delta_x_m = translation_m * np.cos(new_robot_bearing_rad)
        new_robot_delta_y_m = translation_m * np.sin(new_robot_bearing_rad)
        estimate = np.array([new_robot_delta_x_m, new_robot_delta_y_m, rotation_rad])
        if self.debug: print("ESTIMATE DELTAS\n", new_robot_delta_x_m, new_robot_delta_y_m, new_robot_bearing_rad)

        # expand information and vector matrices by one new position
        self.Omega = insert_rows_cols(self.Omega, self.dim, self.dim, self.dim, self.dim)
        self.Xi = insert_rows_cols(self.Xi, self.dim, 0, self.dim, 0)

        # update information matrix/vector based on the robot motion
        for i in range(self.dim * 2):
            self.Omega[i, i] += 1

        for i in range(self.dim):
            if i == 2:  # Psi (orientation) index
                # Give less weight to the motion model for Psi
                self.Omega[self.dim + i, i] += -self.move_weight_psi
                self.Omega[i, self.dim + i] += -self.move_weight_psi
            else:
                # Keep the same weight for X and Y movement
                self.Omega[self.dim + i, i] += -self.move_weight
                self.Omega[i, self.dim + i] += -self.move_weight
            self.Xi[i, 0] += -estimate[i]
            self.Xi[self.dim + i, 0] += estimate[i]


        if self.debug: print("Omega\n", self.Omega)
        if self.debug: print("Xi\n", self.Xi)

        return True

    def map_update( self, motion=True ):
        """
        update map based on processing of class matrices

        Args:
            

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

        # world frame map update
        if self.world_frame_idx != -1:

            # get the world frame state
            self.world_frame_state = self.mu[self.world_frame_idx * self.dim : 
                                             self.world_frame_idx * self.dim + 3]
            if self.debug: print("WORLD FRAME STATE:\n",self.world_frame_state)

            # get an array to offset the current map
            repetitions = self.mu.shape[0] // self.world_frame_state.shape[0]
            self.world_frame_offset = np.tile(self.world_frame_state, (repetitions, 1))

            # update mu based on world frame state
            world_frame_mu = self.mu - self.world_frame_offset
            if self.debug: print("WORLD FRAME MU\n",world_frame_mu)

        # map update
        for key in self.landmarks:
            idx = self.landmarks[key]
            self.map[key][0:2] = self.mu[idx:idx+2, 0]
            angle_norm = normalize_angle(self.map[key][2])
            self.map[key][2] = self.psi_kf.update(angle_norm)
            if self.world_frame_idx != -1:
                self.world_frame_map[key][0:3] = world_frame_mu[idx:idx+3, 0]
                self.world_frame_map[key][2] = normalize_angle(self.world_frame_map[key][2])
                self.map[key][2] = self.world_frame_map[key][2]

        if self.debug: print("map\n", self.map)
        if self.debug: print("world frame map\n",self.world_frame_map)
            
        # reset weights to 1. each
        self.update_weights()

        return True