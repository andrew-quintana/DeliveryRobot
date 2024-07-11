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

Sources:
AI for Games by Ian Millington, Chapter 3 Movement
Robotics {book}: https://www.roboticsbook.org/S52_diffdrive_actions.html

"""
from typing import List, Tuple
from .kinematics_plotting import KinematicsPlotter
import math
import time
import numpy as np
import sys
sys.path.append("../../deliveryrobot")
from utilities.utilities import *
from utilities.computational_geometry import *

def get_unit_vector( vector: np.ndarray ):
    unit_vector = vector / np.linalg.norm(vector)
    return unit_vector

class SteeringOutput( Component ):

    def __init__( self, linear_m_s_2=[0.0,0.0], angular_rad_s_2=0.):
        self.linear_m_s_2 = linear_m_s_2
        self.angular_rad_s_2 = angular_rad_s_2


class Kinematic( Component ):

    def __init__( self,
                 state: np.ndarray,
                 linear_m_s: np.ndarray,
                 rotation_rad_s: float,
                 max_speed_m_s: float,
                 max_turn_rad_s: float ):
        
        position, orientation_rad = state[0:2], state[2]    # convert from [x, y, psi]


        self.kplot = KinematicsPlotter()
        self.position = position                            # [x, y]
        self.orientation_rad = orientation_rad              # float value
        self.linear_m_s = linear_m_s                        # [x', y']
        self.rotation_rad_s = rotation_rad_s
        self.max_speed_m_s = max_speed_m_s
        self.max_turn_rad_s = max_turn_rad_s


    def estimate_update( self, steering: SteeringOutput, time: float ):
        # TODO Implementation: tally time between movement commands and pass as time with last steering command
        # this should make it so it will estimate the relative time since the last move and all subsequent
        # calculations can be made effectively

        if steering == None:
            return
        
        # update the position and orientation
        self.position += self.linear_m_s_2 * time
        self.orientation_rad += self.rotation_rad_s * time

        # update linear_m_s_2 and rotation
        self.linear_m_s_2 += steering.linear_m_s_2 * time
        self.rotation_rad_s += steering.angular_rad_s_2 * time

        # check for speeding and clip
        velocity_magnitude = np.linalg.norm(self.linear_m_s_2)
        if velocity_magnitude > self.max_speed_m_s:
            self.linear_m_s_2 = get_unit_vector(self.linear_m_s_2) * self.max_speed_m_s

        # check for rotating and clip
        if self.rotation_rad_s > self.max_turn_rad_s:
            self.rotation_rad_s = self.max_turn_rad_s

        # create and update plotter
        if self.verbose: 
            self.kplot.pass_kinematic_values(self.time_inputs, self.position_inputs, self.velocity_inputs, self.acceleration_inputs,
                              self.orientation_inputs, self.angular_velocity_inputs, self.angular_acceleration_inputs)

    def slam_update( self, state ):
        # update based on slam feedback
        self.position = state[0:2]
        self.orientation_rad = state[2]

        self.kplot.pass_kinematic_values(slam_time_inputs=time.time(),
                                    slam_position_inputs=self.position,
                                slam_orientation_inputs=self.orientation_rad)

    def new_orientation( self, current: float, velocity: np.ndarray ):
        # ensure velocity
        if np.linalg.norm(velocity) > 0:
            # calculate orientation from the velocity
            return math.atan2(-velocity[0], velocity[1])
        else:
            return current

    def get_velocity_vector( self, linear_m_s_2: np.ndarray, angular_rad_s_2: float, dt: float ):
        # calculate desired velocity vectors
        vx_desired = self.linear_m_s[0] + linear_m_s_2[0] * dt
        vy_desired = self.linear_m_s[1] + linear_m_s_2[1] * dt
        self.linear_m_s = [vx_desired, vy_desired]

        # calculate angle between current velocity and desired velocity
        current_heading = self.orientation_rad
        desired_heading = math.atan2(vy_desired, vx_desired)
        angle_diff = desired_heading - current_heading
        
        print(current_heading)
        print(desired_heading)
        print(angle_diff)

        # calculate desired magnitudes
        velocity_desired = math.sqrt(vx_desired ** 2 + vy_desired ** 2)
        omega_desired = angle_diff / dt
        self.rotation_rad_s = max(min(omega_desired, self.max_turn_rad_s), -self.max_turn_rad_s)

        return velocity_desired, omega_desired
        
    def get_wheel_velocities( self, v: float, omega: float, dt: float, L=0.120, r=0.065 ):
        # based on DDR inverse kinematics equations
        phi_dot_l = (v - (L / 2) * omega) / r
        phi_dot_r = (v + (L / 2) * omega) / r
        
        velocities = np.array([phi_dot_l, phi_dot_r])
        
        if any(filter(lambda x: x > 0.2, velocities)):
            velocities = get_unit_vector(velocities) * 0.2
        
        return phi_dot_l, phi_dot_r
    
    def get_drive_params( self, steering: SteeringOutput, dt:float ):
        velocity, omega = self.get_velocity_vector(steering.linear_m_s_2, steering.angular_rad_s_2, dt)
        print(steering.linear_m_s_2, steering.angular_rad_s_2)
        v_l, v_r = self.get_wheel_velocities(velocity, omega, dt)
        
        
        
        return v_l, v_r

class Path( Component ):
    # hold the path for where the robot will move
    # update as necessary by including new astar paths

    def __init__( self,
                path: List[Tuple[int, np.ndarray]]):
        self.states = [state for _, state in path]
        self.next_idx = 1

    def update_path( self, path: List[Tuple[int, np.ndarray]]):
        self.states = [state for _, state in path]
        self.next_idx = 1

class MovementAI( Component ):
    # TODO once values have been established, set default values for each
    def __init__( self,
                robot: Kinematic,
                target: Kinematic,
                max_acceleration_m_s_2: float,
                max_angular_acceleration_m_s_2: float,
                goal_radius_m: float ):
        self.robot = robot
        self.target = target
        self.max_acceleration_m_s_2 = max_acceleration_m_s_2
        self.max_angular_acceleration_m_s_2 = max_angular_acceleration_m_s_2
        self.goal_radius_m = goal_radius_m
        
        # behavior initialization
        self.arrive = self.Arrive(self, max_speed_m_s=self.robot.max_speed_m_s * 0.5, target_radius_m=0.1, slow_radius_m=0.5)
        self.align = self.Align(self, max_rotation_rad_s=self.robot.max_turn_rad_s, target_thresh_rad=0.1, slow_radius_m=0.5)
        self.path = Path([])
        self.path_following = self.PathFollowing(self, path=self.path, path_point_radius_m=0.2)

    class Seek( Component ):
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            
        def get_steering( self ) -> SteeringOutput:

            result = SteeringOutput()

            # get the direction to the target
            result.linear_m_s_2 = self.outer_instance.target.position - self.outer_instance.robot.position
            
            # the velocity is along this direction, at full speed
            result.linear_m_s_2 = get_unit_vector(result.linear_m_s_2) * self.outer_instance.max_acceleration_m_s_2

            result.angular_rad_s_2 = 0
            return result

    class Arrive( Component ):
        def __init__( self,
                     outer_instance,
                    max_speed_m_s: float,
                    target_radius_m: float,
                    slow_radius_m: float,
                    time_to_target_s: float = 0.1 ):
            self.outer_instance = outer_instance
            self.max_speed_m_s = max_speed_m_s
            self.target_radius_m = target_radius_m
            self.slow_radius_m = slow_radius_m
            self.time_to_target_s = time_to_target_s

            self.max_acceleration_m_s_2 = self.outer_instance.max_acceleration_m_s_2 / 2

        def get_steering( self ) -> SteeringOutput:
            result = SteeringOutput()

            # get direction to target
            direction = self.outer_instance.target.position - self.outer_instance.robot.position
            distance = np.linalg.norm(direction)

            # check if there, return no steering
            if distance < self.target_radius_m:
                return None
            
            # if we are outside the slow_radius_m, then move at max speed
            if distance > self.slow_radius_m:
                target_speed = self.max_speed_m_s
            else:
                target_speed = self.max_speed_m_s * distance / self.slow_radius_m

            # the target velocity combines speed and direction
            target_velocity = direction
            target_velocity = get_unit_vector(target_velocity) * target_speed

            # acceleration tries to get to the target velocity
            result.linear_m_s_2 = target_velocity - self.outer_instance.robot.linear_m_s_2
            result.linear_m_s_2 /= self.time_to_target_s

            # check if the acceleration is too fast
            if result.linear_m_s_2 > self.max_acceleration_m_s_2:
                result.linear_m_s_2 = get_unit_vector(result.linear_m_s_2) * self.max_acceleration_m_s_2

            result.angular = 0
            return result
        
    class Align ( Component ):
        def __init__( self,
                     outer_instance,
                    max_rotation_rad_s: float,
                    target_thresh_rad: float,
                    slow_radius_m: float,
                    time_to_target_s: float = 0.1
                    ):
            self.outer_instance = outer_instance
            self.max_rotation_rad_s = max_rotation_rad_s
            self.target_thresh_rad = target_thresh_rad
            self.slow_radius_m = slow_radius_m
            self.time_to_target_s = time_to_target_s

            self.max_angular_acceleration_m_s_2 = self.outer_instance.max_angular_acceleration_m_s_2 / 2

        def get_steering( self ) -> SteeringOutput:
            result = SteeringOutput()
            
            # get the naive direction to the target
            rotation = self.outer_instance.target.orientation_rad - self.outer_instance.robot.orientation_rad

            # map result to (-pi, pi) interval
            rotation = normalize_angle(rotation)
            rotation_size = abs(rotation)

            # check if we are there, return no steering
            if rotation_size < self.target_thresh_rad:
                return None
            
            # if we are outside the slow_radius_m, then use maximum rotation
            if rotation_size > self.slow_radius_m:
                target_rotation = self.max_rotation_rad_s

            # calculate scaled rotation
            else:
                target_rotation = self.max_rotation_rad_s * rotation_size / self.slow_radius_m

            # the final target rotation combines speed (variable) and direction
            target_rotation *= rotation / rotation_size

            # acceleration tries to get to the target rotation
            angular_acceleration = abs(result.angular_rad_s_2)
            if angular_acceleration > self.max_angular_acceleration_m_s_2:
                result.angular_rad_s_2 /= angular_acceleration    # normalize, maintaining sign
                result.angular_rad_s_2 *= self.max_angular_acceleration_m_s_2

            result.linear_m_s_2 = 0
            return result
        


    class PathFollowing( Seek ):
        def __init__( self,
                     outer_instance,
                    path: Path, 
                    path_point_radius_m: float):
            super().__init__(outer_instance)
            self.path_point_radius_m = path_point_radius_m

        def get_steering( self ) -> SteeringOutput:
            # check if position is close enough to goal
            distance_to_goal_m = abs(self.outer_instance.target.position - self.outer_instance.robot.position)
            if np.linalg.norm(distance_to_goal_m) < self.path_point_radius_m:
                return None
            
            # check if position is close enough to next path point
            distance_to_target_m = abs(self.outer_instance.path.states[self.outer_instance.path.next_idx] - self.outer_instance.robot.position)
            if np.linalg.norm(distance_to_target_m) < self.path_point_radius_m:
                self.outer_instance.path.next_index += 1

            # set next path point as target position
            self.outer_instance.target.position = self.outer_instance.path.states[self.outer_instance.path.next_idx]

            # delegate to seek
            return super().get_steering()