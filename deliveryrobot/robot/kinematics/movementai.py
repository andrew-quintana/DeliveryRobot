"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Estimate dead reckoning and compute acceleration vectors for entity based on targets.

Classes:
    SteeringOutput: accleration information
    Kinematic: dynamic entity to be tracked with dead reckoning
        get_velocity_vector(): get and update velocity for kinematic object
        get_wheel_velocities(): calculate wheel velocities based on desired linear and
            radial velocity
        bias_correction(): calculator for wheel biases
        get_drive_params(): execute steps involved in getting the drive parameters
        estimate_distance_traveled(): estimate how far the entity traveled since the
            last function call
        estimate_update(): estimate the dead reckoning state
        slam_update(): method of contributing external mapping information for localization
    Path: path information
        update_path(): update the path and prepare for path following functionality
    MovementAI: higher level acceleration vector determination
        Arrive: approach the target and slow down to avoid missing
        Align: change the orientation to correspond with one desired
        Seek: move toward a point in space
        PathFollowing: move toward a sequence of points in space
            - extends Seek

Functions:
    get_unit_vector(): calculate the unit vector

Dependencies:
    utilities.py
    computational_geometry.py
    kinematics_plotting.py

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
#from .pid import PIDController
import math
import time
import numpy as np
import sys
sys.path.append("../../deliveryrobot")
from utilities.utilities import *
from utilities.computational_geometry import *
import logging
import os
from datetime import datetime

# Create the directory if it does not exist
os.makedirs('test_output', exist_ok=True)

# Generate the timestamped filename
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
log_filename = f'test_output/{timestamp}.txt'

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_unit_vector( vector: np.ndarray ):
    unit_vector = vector / np.linalg.norm(vector)
    return unit_vector

class SteeringOutput( Component ):

    def __init__( self, linear_m_s_2=np.array([0.0,0.0],dtype=np.float64), angular_rad_s_2=0.):
        self.linear_m_s_2 = linear_m_s_2
        self.angular_rad_s_2 = angular_rad_s_2


class Kinematic( Component ):

    def __init__( self,
                 state: np.ndarray,
                 linear_m_s: np.ndarray,
                 rotation_rad_s: float,
                 max_speed_m_s: float,
                 max_turn_rad_s: float ):
        
        position, orientation_rad = state[0:2].astype(np.float64), state[2]    # convert from [x, y, psi]

        self.kplot = KinematicsPlotter()
        
        # kinematics variables
        self.position = position                            # [x, y]
        self.orientation_rad = orientation_rad              # float value
        self.linear_m_s = linear_m_s.astype(np.float64)                        # [x', y']
        self.wheel_velocities = np.array([0.,0.],dtype=np.float64)
        self.rotation_rad_s = rotation_rad_s
        
        # boundary conditions
        self.max_speed_m_s = max_speed_m_s
        self.max_turn_rad_s = max_turn_rad_s
        self.min_wheel_velocity_m_s = 0.05
        
        # control memory
        self.steering = SteeringOutput()
        self.last_call = time.time()                        # timestamp representing last estimated state

    def new_orientation( self, current: float, velocity: np.ndarray ):
        # ensure velocity
        if np.linalg.norm(velocity) > 0:
            # calculate orientation from the velocity
            return math.atan2(-velocity[0], velocity[1])
        else:
            return current

    def get_velocity_vector(self, linear_m_s_2: np.ndarray, angular_rad_s_2: float, dt: float):
        # get and update velocity for kinematic object
        
        # calculate desired velocity vectors
        vx_desired = self.linear_m_s[0] + linear_m_s_2[0] * dt
        vy_desired = self.linear_m_s[1] + linear_m_s_2[1] * dt
        
        # bottom limit
        if vx_desired < 1e-5: vx_desired = 0
        if vy_desired < 1e-5: vy_desired = 0
        
        # check for speeding and clip
        velocity = np.array([vx_desired, vy_desired], dtype=np.float64)
        velocity_magnitude = np.linalg.norm(velocity)
    
        logging.debug(f"{velocity_magnitude} > {self.max_speed_m_s}")
        # update to capped speed if over maximum velocity
        if velocity_magnitude > self.max_speed_m_s:
            self.linear_m_s = get_unit_vector(velocity) * self.max_speed_m_s
        else:
            self.linear_m_s = np.array([vx_desired, vy_desired], dtype=np.float64)
            
        logging.debug(f"chosen velocities: {self.linear_m_s}")
        velocity_desired = math.sqrt(vx_desired ** 2 + vy_desired ** 2)
        
        # Consider angular acceleration
        self.rotation_rad_s += angular_rad_s_2 * dt
        self.rotation_rad_s = max(min(self.rotation_rad_s, self.max_turn_rad_s), -self.max_turn_rad_s)
                
        logging.debug(f"omega: {self.rotation_rad_s}")

        return velocity_desired, self.rotation_rad_s
        
    def get_wheel_velocities( self, v: float, omega: float, L=0.120, r=0.065 ):
        # based on DDR inverse kinematics equations
        phi_dot_l = (v - (L / 2) * omega) / r
        phi_dot_r = (v + (L / 2) * omega) / r
        
        self.wheel_velocities = np.array([phi_dot_l, phi_dot_r])
        
        return self.wheel_velocities
    
    def bias_correction( self, v_l, v_r ):
        
        v_r *= 1.03
        
        return v_l, v_r
    
    def get_drive_params( self, steering: SteeringOutput, dt: float ):
        velocity, omega = self.get_velocity_vector(steering.linear_m_s_2, steering.angular_rad_s_2, dt)
        v_l, v_r = self.get_wheel_velocities(velocity, omega)
        #v_l, v_r = self.bias_correction(v_l, v_r)
        
        logging.debug(f"wheel velocities: {v_l}, {v_r}")
        
        return v_l, v_r
    
    def estimate_distance_traveled(self, dt: float, L=0.120):
        v_l, v_r = self.wheel_velocities

        # Calculate linear and angular velocities
        v = (v_r + v_l) / 2
        omega = (v_r - v_l) / L

        # Calculate the change in bearing
        delta_theta = omega * dt

        # Print for debugging
        logging.debug(f"v_l: {v_l}, v_r: {v_r}, v: {v}, omega: {omega}, delta_theta: {delta_theta}")

        # Estimate the distance traveled
        if abs(omega) < 1e-6:
            # Straight movement
            delta_x = v * dt * np.cos(self.orientation_rad)
            delta_y = v * dt * np.sin(self.orientation_rad)
        else:
            # Circular movement
            radius = v / omega
            delta_x = radius * (np.sin(self.orientation_rad + delta_theta) - np.sin(self.orientation_rad))
            delta_y = radius * (np.cos(self.orientation_rad) - np.cos(self.orientation_rad + delta_theta))
            
        return delta_x, delta_y, delta_theta
    
    def estimate_update( self, call_time: float ):
        # TODO Implementation: tally time between movement commands and pass as time with last steering command
        # this should make it so it will estimate the relative time since the last move and all subsequent
        # calculations can be made effectively

        if self.steering == None:
            return
        
        # get dt from last call
        dt = call_time
        
        # estimate the distances traveled
        delta_x, delta_y, delta_theta = self.estimate_distance_traveled(dt)
        
        self.position += np.array([delta_x, delta_y]) * (0.27/0.2)   # with estimated error ratio (opportunity for RL)
        self.orientation_rad += delta_theta * (0.2/0.22)

        # Additional logging statements
        logging.debug(f"distances: {delta_x}, {delta_y} over {dt}")
        logging.debug(f"orientation: {delta_theta} over {dt}")
        logging.debug("\n- - - - - - - - - - -\n")
        logging.debug(f"Time since last call: {dt}")
        logging.debug(f"Steering: {self.steering.linear_m_s_2}, {self.steering.angular_rad_s_2}")
        logging.debug(f"Position: {self.position}")
        logging.debug(f"Orientation: {self.orientation_rad}")
        logging.debug(f"Velocity: {self.linear_m_s}")
        logging.debug(f"Rotation Vel: {self.rotation_rad_s}")
        
        self.last_call = time.time()

        # create and update plotter
        if True: 
            
            self.kplot.add_data_point(
                time=time.time(),
                position=self.position,
                orientation=self.orientation_rad,
                velocity=self.linear_m_s,
                angular_velocity=self.rotation_rad_s,
                acceleration=self.steering.linear_m_s_2,
                angular_acceleration=self.steering.angular_rad_s_2)
            
        return delta_x, delta_y, delta_theta

    def slam_update( self, state ):
        # update based on slam feedback
        self.position = state[0:2]
        self.orientation_rad = state[2]

        self.kplot.add_data_point(
                time=time.time(),
                position=self.position,
                orientation=self.orientation_rad)
        
        # set a new last estimated timestamp
        self.last_call = time.time()

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
        
        # entities
        self.robot = robot
        self.target = target
        
        # boundary conditions
        self.max_acceleration_m_s_2 = max_acceleration_m_s_2
        self.max_angular_acceleration_m_s_2 = max_angular_acceleration_m_s_2
        self.goal_radius_m = goal_radius_m
        
        # behavior initialization
        self.arrive = self.Arrive(self, max_speed_m_s=self.robot.max_speed_m_s, target_radius_m=0.005, slow_radius_m=0.05)
        self.align = self.Align(self, max_rotation_rad_s=self.robot.max_turn_rad_s, target_thresh_rad=0.1, slow_radius_m=0.5)
        self.path = Path([])
        self.path_following = self.PathFollowing(self, path_point_radius_m=0.2)
        
        """# PID controllers
        self.linear_pid_controller = PIDController(
            tau_p=1.0,
            tau_d=0.1,
            tau_i=0.01,
            tune_with_twiddle=True,
            target_function=self.target_function)
        self.angular_pid_controller = PIDController(tau_p=1.0, tau_d=0.1, tau_i=0.01, tune_with_twiddle=True, target_function=self.target_function)"""
        
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

        def get_steering( self, call_time:float ) -> SteeringOutput:
            result = SteeringOutput()

            # update estimate of position
            delta_x, delta_y, delta_theta = self.outer_instance.robot.estimate_update( call_time )

            # get direction to target
            direction = self.outer_instance.target.position - self.outer_instance.robot.position
            distance = np.linalg.norm(direction)

            # check if there, return no steering
            if distance < self.target_radius_m:
                logging.debug("---------HERE---------")
<<<<<<< HEAD
                return [None, delta_x, delta_y, delta_theta]
=======
                return None
>>>>>>> 8390973c0df02bf66cf1a728126322ce61e9c615
            
            # if we are outside the slow_radius_m, then move at max speed
            if distance > self.slow_radius_m:
                target_speed = self.max_speed_m_s
            else:
                target_speed = self.max_speed_m_s * distance / self.slow_radius_m

            # the target velocity combines speed and direction
            target_velocity = direction
            target_velocity = get_unit_vector(target_velocity) * target_speed
            
            logging.debug(f"target_velocity {target_velocity}")

            # acceleration tries to get to the target velocity
            result.linear_m_s_2 = target_velocity - self.outer_instance.robot.linear_m_s
            result.linear_m_s_2 /= self.time_to_target_s

            # check if the acceleration is too fast
            if np.linalg.norm(result.linear_m_s_2) > self.max_acceleration_m_s_2:
                result.linear_m_s_2 = get_unit_vector(result.linear_m_s_2) * self.max_acceleration_m_s_2
            
            # Induced Angular Acceleration
            target_orientation = math.atan2(direction[1], direction[0])
            rotation = target_orientation - self.outer_instance.robot.orientation_rad
            rotation = normalize_angle(rotation)  # normalize to [-pi, pi]
            
            # Proportional control for angular velocity
            result.angular_rad_s_2 = rotation - self.outer_instance.robot.rotation_rad_s            
            if abs(result.angular_rad_s_2) > self.outer_instance.max_angular_acceleration_m_s_2:
                result.angular_rad_s_2 = math.copysign(
                    self.outer_instance.max_angular_acceleration_m_s_2,
                    result.angular_rad_s_2)

            self.outer_instance.robot.steering = result

            return [result, delta_x, delta_y, delta_theta]
        
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

        def get_steering( self, call_time:float ) -> SteeringOutput:
            result = SteeringOutput()

            # update estimate of position
            delta_x, delta_y, delta_theta = self.outer_instance.robot.estimate_update( call_time )
            
            # get the naive direction to the target
            rotation = self.outer_instance.target.orientation_rad - self.outer_instance.robot.orientation_rad

            # map result to (-pi, pi) interval
            rotation = normalize_angle(rotation)
            rotation_size = abs(rotation)

            # check if we are there, return no steering
            if rotation_size < self.target_thresh_rad:
                return [None, delta_x, delta_y, delta_theta]
            
            # if we are outside the slow_radius_m, then use maximum rotation
            if rotation_size > self.slow_radius_m:
                target_rotation = self.max_rotation_rad_s

            # calculate scaled rotation
            else:
                target_rotation = self.max_rotation_rad_s * rotation_size / self.slow_radius_m

            # the final target rotation combines speed (variable) and direction
            target_rotation *= rotation / rotation_size
            
            # acceleration tries to get to the target rotation
            result.angular_rad_s_2 = target_rotation - self.outer_instance.robot.rotation_rad_s
            result.angular_rad_s_2 /= self.time_to_target_s
            
            # check if acceleration above threshold
            angular_acceleration = abs(result.angular_rad_s_2)
            if angular_acceleration > self.outer_instance.max_angular_acceleration_m_s_2:
                result.angular_rad_s_2 /= angular_acceleration    # normalize, maintaining sign
                result.angular_rad_s_2 *= self.outer_instance.max_angular_acceleration_m_s_2

            result.linear_m_s_2 = [0.,0.]

            self.outer_instance.robot.steering = result

            return [result, delta_x, delta_y, delta_theta]
        
    class Seek( Component ):
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            
        def get_steering( self, delta_x, delta_y, delta_theta ) -> SteeringOutput:

            result = SteeringOutput()

            # get the direction to the target
            result.linear_m_s_2 = self.outer_instance.target.position - self.outer_instance.robot.position
            
            # the velocity is along this direction, at full speed
            result.linear_m_s_2 = get_unit_vector(result.linear_m_s_2) * self.outer_instance.max_acceleration_m_s_2

            # Induced Angular Acceleration
            target_orientation = math.atan2(result.linear_m_s_2[1], result.linear_m_s_2[0])
            rotation = target_orientation - self.outer_instance.robot.orientation_rad
            rotation = normalize_angle(rotation)  # normalize to [-pi, pi]
            
            # Proportional control for angular velocity
            result.angular_rad_s_2 = rotation - self.outer_instance.robot.rotation_rad_s            
            if abs(result.angular_rad_s_2) > self.outer_instance.max_angular_acceleration_m_s_2:
                result.angular_rad_s_2 = math.copysign(
                    self.outer_instance.max_angular_acceleration_m_s_2,
                    result.angular_rad_s_2)
            
            self.outer_instance.robot.steering = result
            
            return [result, delta_x, delta_y, delta_theta]
        
    class Seek( Component ):
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            
        def get_steering( self ) -> SteeringOutput:

            result = SteeringOutput()

            # get the direction to the target
            result.linear_m_s_2 = self.outer_instance.target.position - self.outer_instance.robot.position
            
            # the velocity is along this direction, at full speed
            result.linear_m_s_2 = get_unit_vector(result.linear_m_s_2) * self.outer_instance.max_acceleration_m_s_2

            # Induced Angular Acceleration
            target_orientation = math.atan2(result.linear_m_s_2[1], result.linear_m_s_2[0])
            rotation = target_orientation - self.outer_instance.robot.orientation_rad
            rotation = normalize_angle(rotation)  # normalize to [-pi, pi]
            
            # Proportional control for angular velocity
            result.angular_rad_s_2 = rotation - self.outer_instance.robot.rotation_rad_s            
            if abs(result.angular_rad_s_2) > self.outer_instance.max_angular_acceleration_m_s_2:
                result.angular_rad_s_2 = math.copysign(
                    self.outer_instance.max_angular_acceleration_m_s_2,
                    result.angular_rad_s_2)
            
            self.outer_instance.robot.steering = result
            
            return result
        
    class PathFollowing( Seek ):
        def __init__( self,
                     outer_instance,
                    path_point_radius_m: float):
            super().__init__(outer_instance)
            self.path_point_radius_m = path_point_radius_m

        def get_steering( self, call_time:float ) -> SteeringOutput:
            # TODO FSM Design: once path is going for path[-1], set target to goal_state
            
            # update estimate of position
<<<<<<< HEAD
            delta_x, delta_y, delta_theta = self.outer_instance.robot.estimate_update(call_time)
=======
            self.outer_instance.robot.estimate_update(call_time)
>>>>>>> 8390973c0df02bf66cf1a728126322ce61e9c615
            
            # check if position is close enough to next path point
            next_state = self.outer_instance.path.states[self.outer_instance.path.next_idx]
            logging.debug(next_state)
            distance_to_target_m = abs(next_state[:2] - self.outer_instance.robot.position)
            if np.linalg.norm(distance_to_target_m) < self.path_point_radius_m:
                self.outer_instance.path.next_idx += 1
                
            # check if the next path node is the final one, report completion
            if self.outer_instance.path.next_idx == len(self.outer_instance.path.states) - 1:
                # do not set next path point to avoid combining A* and MovementAI goal tolerances
<<<<<<< HEAD
                return [None, delta_x, delta_y, delta_theta]
=======
                return None
>>>>>>> 8390973c0df02bf66cf1a728126322ce61e9c615

            # set next path point as target position
            self.outer_instance.target.position = self.outer_instance.path.states[self.outer_instance.path.next_idx][0:2]
            self.outer_instance.target.orientation_rad = self.outer_instance.path.states[self.outer_instance.path.next_idx][2]

            # delegate to seek
            output = super().get_steering(delta_x, delta_y, delta_theta)
            print(output)
            return output