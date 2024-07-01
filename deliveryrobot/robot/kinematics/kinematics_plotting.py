import matplotlib.pyplot as plt
from IPython.display import clear_output

class KinematicsPlotter:
    
    def __init__(self):
        # Initialize empty lists to store values
        self.time_values = []
        self.position_values = []
        self.orientation_values = []
        self.velocity_values = []
        self.angular_velocity_values = []
        self.acceleration_values = []
        self.angular_acceleration_values = []
        self.slam_position_values = []  # SLAM position measurements
        self.slam_orientation_values = []  # SLAM orientation measurements
        self.slam_time_values = []  # SLAM timestamps

        # Initialize the plots and lines
        self.fig, self.axs = plt.subplots(2, 3, figsize=(20, 10))  # 2 rows, 3 columns
        self.line1, = self.axs[0, 0].plot([], [])  # Position x
        self.line2, = self.axs[0, 0].plot([], [])  # Position y
        self.line3, = self.axs[0, 1].plot([], [])  # Velocity
        self.line4, = self.axs[0, 2].plot([], [])  # Acceleration
        self.line5, = self.axs[1, 0].plot([], [])  # Orientation
        self.line6, = self.axs[1, 1].plot([], [])  # Angular Velocity
        self.line7, = self.axs[1, 2].plot([], [])  # Angular Acceleration
        self.line8, = self.axs[0, 0].plot([], [], 'ro')  # SLAM position x
        self.line9, = self.axs[0, 0].plot([], [], 'ro')  # SLAM position y
        self.line10, = self.axs[1, 0].plot([], [], 'ro')  # SLAM orientation

    def pass_kinematic_values(self, time_inputs=None, position_inputs=None, velocity_inputs=None, acceleration_inputs=None,
                              orientation_inputs=None, angular_velocity_inputs=None, angular_acceleration_inputs=None,
                              slam_time_inputs=None, slam_position_inputs=None, slam_orientation_inputs=None):
        if time_inputs is not None:
            self.time_values.append(time_inputs)
        if position_inputs is not None:
            self.position_values.append(position_inputs)
        if orientation_inputs is not None:
            self.orientation_values.append(orientation_inputs)
        if velocity_inputs is not None:
            self.velocity_values.append(velocity_inputs)
        if angular_velocity_inputs is not None:
            self.angular_velocity_values.append(angular_velocity_inputs)
        if acceleration_inputs is not None:
            self.acceleration_values.append(acceleration_inputs)
        if angular_acceleration_inputs is not None:
            self.angular_acceleration_values.append(angular_acceleration_inputs)
        if slam_time_inputs is not None:
            self.slam_time_values.append(slam_time_inputs)
        if slam_position_inputs is not None:
            self.slam_position_values.append(slam_position_inputs)
        if slam_orientation_inputs is not None:
            self.slam_orientation_values.append(slam_orientation_inputs)

    def update_plots(self):
        # Clear the previous plot
        clear_output(wait=True)

        # Update data
        if self.position_values:
            self.line1.set_data(self.time_values, [pos[0] for pos in self.position_values])  # Position x
            self.line2.set_data(self.time_values, [pos[1] for pos in self.position_values])  # Position y
        if self.velocity_values:
            self.line3.set_data(self.time_values, self.velocity_values)
        if self.acceleration_values:
            self.line4.set_data(self.time_values, self.acceleration_values)
        if self.orientation_values:
            self.line5.set_data(self.time_values, self.orientation_values)
        if self.angular_velocity_values:
            self.line6.set_data(self.time_values, self.angular_velocity_values)
        if self.angular_acceleration_values:
            self.line7.set_data(self.time_values, self.angular_acceleration_values)
        if self.slam_position_values:
            self.line8.set_data(self.slam_time_values, [pos[0] for pos in self.slam_position_values])  # SLAM position x
            self.line9.set_data(self.slam_time_values, [pos[1] for pos in self.slam_position_values])  # SLAM position y
        if self.slam_orientation_values:
            self.line10.set_data(self.slam_time_values, self.slam_orientation_values)

        # Adjust axes limits
        for ax, values in zip(self.axs.flatten(), [self.position_values, self.velocity_values, self.acceleration_values,
                                                   self.orientation_values, self.angular_velocity_values, self.angular_acceleration_values,
                                                   self.slam_position_values, self.slam_orientation_values]):
            ax.relim()
            ax.autoscale_view()

        # Set titles and labels
        self.axs[0, 0].set_title('Position vs Time')
        self.axs[0, 0].set_xlabel('Time')
        self.axs[0, 0].set_ylabel('Position')

        self.axs[0, 1].set_title('Velocity vs Time')
        self.axs[0, 1].set_xlabel('Time')
        self.axs[0, 1].set_ylabel('Velocity')

        self.axs[0, 2].set_title('Acceleration vs Time')
        self.axs[0, 2].set_xlabel('Time')
        self.axs[0, 2].set_ylabel('Acceleration')

        self.axs[1, 0].set_title('Orientation vs Time')
        self.axs[1, 0].set_xlabel('Time')
        self.axs[1, 0].set_ylabel('Orientation')

        self.axs[1, 1].set_title('Angular Velocity vs Time')
        self.axs[1, 1].set_xlabel('Time')
        self.axs[1, 1].set_ylabel('Angular Velocity')

        self.axs[1, 2].set_title('Angular Acceleration vs Time')
        self.axs[1, 2].set_xlabel('Time')
        self.axs[1, 2].set_ylabel('Angular Acceleration')

        # Redraw the plot
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # pause for a short period to allow the plot to update