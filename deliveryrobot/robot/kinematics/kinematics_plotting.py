"""

Author: Andrew Quintana
Email: aquintana7@gatech.edu
Version: 0.1.0
License: [License Name]

Usage:
Pass the KinematicsPlotter your iterative state updates to create visual depictions of your system's movement.

Classes & Functions:
    KinematicsPlotter: enables ability to save and update kinematics data for plotting
        add_data_point(): add data ponts from each iteration to build datasets for plotting
        plot_data(): plot iterative state information
        clear_data(): reset data and start again

Dependencies:
    utilities.py

License:
[Include the full text of the license you have chosen for your code]

Examples:
[Provide some example code snippets demonstrating how to use the module/package]

Sources:

"""

import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../deliveryrobot")
from utilities.utilities import *

class KinematicsPlotter( Component ):
    
    def __init__(self):
        # Initialize empty lists to store values
        self.time_values = []
        self.position_values = []
        self.orientation_values = []
        self.velocity_values = []
        self.angular_velocity_values = []
        self.acceleration_values = []
        self.angular_acceleration_values = []

    def add_data_point(self, time, position=None, orientation=None, velocity=None, angular_velocity=None, acceleration=None, angular_acceleration=None):
        self.time_values.append(time)
        self.position_values.append(np.copy(position) if position is not None else (None, None))
        self.orientation_values.append(orientation if orientation is not None else None)
        self.velocity_values.append(np.copy(velocity) if velocity is not None else (None, None))
        self.angular_velocity_values.append(angular_velocity)
        self.acceleration_values.append(np.copy(acceleration) if acceleration is not None else (None, None))
        self.angular_acceleration_values.append(angular_acceleration)
        
        # Debug prints to check the data being added
        print(f"Added data point: time={time}, position={position}, orientation={orientation}, velocity={velocity}, angular_velocity={angular_velocity}, acceleration={acceleration}, angular_acceleration={angular_acceleration}")

    def plot_data(self):
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 3 columns
        
        # Set titles for the plots
        axs[0, 0].set_title("Position")
        axs[0, 1].set_title("Velocity")
        axs[0, 2].set_title("Acceleration")
        axs[1, 0].set_title("Orientation")
        axs[1, 1].set_title("Angular Velocity")
        axs[1, 2].set_title("Angular Acceleration")

        # Set x and y labels
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Position")
        axs[0, 1].set_xlabel("Time")
        axs[0, 1].set_ylabel("Velocity")
        axs[0, 2].set_xlabel("Time")
        axs[0, 2].set_ylabel("Acceleration")
        axs[1, 0].set_xlabel("Time")
        axs[1, 0].set_ylabel("Orientation")
        axs[1, 1].set_xlabel("Time")
        axs[1, 1].set_ylabel("Angular Velocity")
        axs[1, 2].set_xlabel("Time")
        axs[1, 2].set_ylabel("Angular Acceleration")

        # Plot position
        time_arr = np.array(self.time_values)
        position_arr = np.array(self.position_values)
        velocity_arr = np.array(self.velocity_values)
        acceleration_arr = np.array(self.acceleration_values)
        
        axs[0, 0].plot(time_arr, position_arr[:, 0], label="Position x", color='blue')
        axs[0, 0].plot(time_arr, position_arr[:, 1], label="Position y", color='green')
        
        # Plot velocity
        axs[0, 1].plot(time_arr, velocity_arr[:, 0], label="Velocity x", color='blue')
        axs[0, 1].plot(time_arr, velocity_arr[:, 1], label="Velocity y", color='green')
        
        # Plot acceleration
        axs[0, 2].plot(time_arr, acceleration_arr[:, 0], label="Acceleration x", color='blue')
        axs[0, 2].plot(time_arr, acceleration_arr[:, 1], label="Acceleration y", color='green')
        
        # Plot orientation
        axs[1, 0].plot(time_arr, self.orientation_values, label="Orientation", color='blue')
        
        # Plot angular velocity
        axs[1, 1].plot(time_arr, self.angular_velocity_values, label="Angular Velocity", color='blue')
        
        # Plot angular acceleration
        axs[1, 2].plot(time_arr, self.angular_acceleration_values, label="Angular Acceleration", color='blue')

        # Add legends
        for ax in axs.flat:
            ax.legend()

        plt.tight_layout()
        plt.show()

    def clear_data(self):
        self.time_values.clear()
        self.position_values.clear()
        self.orientation_values.clear()
        self.velocity_values.clear()
        self.angular_velocity_values.clear()
        self.acceleration_values.clear()
        self.angular_acceleration_values.clear()