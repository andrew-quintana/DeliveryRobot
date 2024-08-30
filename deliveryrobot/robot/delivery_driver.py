"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Setup the Jetbot and prepare movement functionality.

Classes:
    DeliveryRobot: all functionality for the jetbot for this project's utility.
        take_picture(): take an image and return its location
        stop(): stop the robot's movement
        step_forward(): step forward a specific amount of time
        step_backward(): step backward a specific amount of time
        step_left(): step left a specific amount of time
        step_right(): step right a specific amount of time
        slam_update_ai(): supports updating state information for acceleration vector
            calculating algorithms
        update_path_ai(): updates path for path following movement
        path_follow_ai(): path following movement execution
        arrive_ai(): arrival movement execution
        align_ai(): align movement execution

Dependencies:
    movementai.py
    utilities.py
    jetbot
        Robot.py
        Camera.py

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

"""


import platform
import sys
import os


from kinematics.movementai import *

sys.path.append("../../../jetbot/")
from jetbot import Robot, Camera, bgr8_to_jpeg

sys.path.append("../../deliveryrobot")
from utilities.utilities import *

from uuid import uuid1
import cv2
import os
import time
import numpy as np
from typing import List, Tuple

class DeliveryRobot:
    def __init__(self):

        # hardware setup
        self.robot = Robot()
        self.camera = Camera.instance(width=1224, height=1224)

        # AI movement setup
        self.robot_ai = Kinematic(
            state=np.array([0, 0, 0], dtype=np.float64),
            linear_m_s=np.array([0, 0], dtype=np.float64),
            rotation_rad_s=0.0,
            max_speed_m_s=0.015,
            max_turn_rad_s=0.1
        )
        self.target_ai = Kinematic(
            state=np.array([0, 0, 0], dtype=np.float64),
            linear_m_s=np.array([0, 0], dtype=np.float64),
            rotation_rad_s=0.0,
            max_speed_m_s=0.0,
            max_turn_rad_s=0.0
        )
        self.movement_ai = MovementAI(
            self.robot_ai,
            self.target_ai,
            max_acceleration_m_s_2=0.01,
            max_angular_acceleration_m_s_2=0.1,
            goal_radius_m=0.05)
        

    def take_picture(self, directory, filename=""):
        if filename == "": name = str(uuid1()) + '.jpg'
        else: name = filename
        filename = os.path.join(directory, name)
        cv2.imwrite(filename, self.camera.value)
        return filename

    def stop(self):
        self.robot.stop()

    def step_forward(self, speed_m_s=0.4, time_s=0.5):
        self.robot.forward(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_backward(self, speed_m_s=0.4, time_s=0.5):
        self.robot.backward(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_left(self, speed_m_s=0.4, time_s=0.5):
        self.robot.left(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def step_right(self, speed_m_s=0.4, time_s=0.5):
        self.robot.right(speed_m_s)
        time.sleep(time_s)
        self.robot.stop()

    def slam_update_ai(self, robot_state: np.ndarray, goal_state: np.ndarray):
        # update based on slam states
        self.robot_ai.slam_update(robot_state)
        self.target_ai.slam_update(goal_state)

    def update_path_ai(self, path: List[Tuple[int, np.ndarray]]):
        # check if created
        self.movement_ai.path.update_path(path)

    def path_follow_ai(self, dt: float, call_time:float ):
        
        # get steering command
        steering, delta_x, delta_y, delta_theta = self.movement_ai.path_following.get_steering( call_time )
        
        # test for finish
        if steering == None:
            print("PATH COMPLETE, ARRIVING")
            return [0, delta_x, delta_y, delta_theta]
        else:
            print(time.time(),"steering", steering.linear_m_s_2, steering.angular_rad_s_2)
        
        # get and update drive parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)
        
        # actuate motors
        self.robot.set_motors(v_left, v_right)
        
        self.robot_ai.last_call = time.time()
        
        return [1, delta_x, delta_y, delta_theta]

    def arrive_ai(self, dt:float, call_time:float):

        # get steering command
        steering, delta_x, delta_y, delta_theta = self.movement_ai.arrive.get_steering( call_time )
        
        # test for finish
        if steering == None:
            print("ARRIVED")
            return [0, delta_x, delta_y, delta_theta]
        else:
            print(time.time(),"steering", steering.linear_m_s_2, steering.angular_rad_s_2)
        
        # get and update drive parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)
        
        # actuate motors
        self.robot.set_motors(v_left, v_right)
        
        self.robot_ai.last_call = time.time()
        
        return [1, delta_x, delta_y, delta_theta]

    def align_ai(self, dt:float, call_time:float):
        
        # get steering command
        steering, delta_x, delta_y, delta_theta = self.movement_ai.align.get_steering( call_time )
        
        # test for finish
        if steering == None:
            print("ALIGNED")
            return [0, delta_x, delta_y, delta_theta]
        else:
            print(time.time(),"steering", steering.linear_m_s_2, steering.angular_rad_s_2)

        # determine driving parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)

        # actuate motors
        self.robot.set_motors(v_left, v_right)
        
        self.robot_ai.last_call = time.time()
        
        return [1, delta_x, delta_y, delta_theta]