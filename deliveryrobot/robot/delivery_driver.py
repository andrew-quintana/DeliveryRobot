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


import platform
import sys
import os

sys.path.append("../../../jetbot/")
sys.path.append("../../deliveryrobot")

from jetbot import Robot, Camera, bgr8_to_jpeg
from utilities.utilities import *
from deliveryrobot.robot.movementai import *

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
        self.camera = Camera.instance(width=1224, height=1124)

        # AI movement setup
        self.robot_ai = Kinematic(
            state=np.array([0,0,0]),
            linear_m_s_2=np.array([0,0]),
            rotation_rad_s=0.0,
            max_speed_m_s=0.5,
            max_turn_rad_s=1.0)
        self.target_ai = Kinematic(
            state=np.array([0,0,0]),
            linear_m_s_2=np.array([0,0]),
            rotation_rad_s=0.0,
            max_speed_m_s=0.0,
            max_turn_rad_s=0.0)
        self.movement_ai = MovementAI(
            self.robot_ai,
            self.target_ai,
            max_acceleration_m_s_2=0.5,
            max_angular_acceleration_m_s_2=1.0,
            goal_radius_m=0.1)


    def take_picture(self, directory):
        filename = os.path.join(directory, str(uuid1()) + '.jpg')
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

    def update_path_ai(self, path: List[Tuple[int, np.ndarray]], goal_idx: int):
        # check if created
        if hasattr(self, 'path'):
            self.path.update_path(path)
        else:
            self.path = Path(path, goal_idx)

    def path_follow_ai(self, dt: float):
        # determine steering commands
        steering = self.movement_ai.PathFollowing(path=self.path, path_point_radius_m=0.050)

        # determine driving parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)

        # actuate motors
        self.robot.left(v_left)
        self.robot.right(v_right)

    def arrive_ai(self, dt: float):
        # determine steering commands
        steering = self.movement_ai.Arrive(
            max_speed_m_s=0.5,
            target_radius_m=0.010,
            slow_radius_m=0.050,
            time_to_target_s=0.5
        )

        # determine driving parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)

        # actuate motors
        self.robot.left(v_left)
        self.robot.right(v_right)

    def align_ai(self, dt: float):
        # determine steering commands
        steering = self.movement_ai.Align(
            max_rotation_rad_s=0.3,
            target_thresh_rad=0.05,
            slow_radius_m=0.5,
            time_to_target_s=0.5
        )

        # determine driving parameters
        v_left, v_right = self.robot_ai.get_drive_params(steering, dt)

        # actuate motors
        self.robot.left(v_left)
        self.robot.right(v_right)