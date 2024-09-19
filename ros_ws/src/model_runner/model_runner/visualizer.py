import torch
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles
from robo_limb_rl.arch.Q_net import QNet_MLP
from matplotlib import pyplot as plt

class Visualizer(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('visualizer')
        # Declare parameters using declare_parameters

        # Retrieve the parameters
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('X Angle (degrees)')
        self.ax.set_ylabel('Y Angle (degrees)')
        self.ax.set_title('Live State Visualization')
        self.current_x, self.current_y = [], []
        self.line, = self.ax.plot(self.current_x, self.current_y, 'b-', markersize=10)
        self.scatter = self.ax.scatter([], [], c='purple', s=100)
        plt.draw()
        plt.pause(0.0001)
        
    # stores the newest received angles
    def angle_listener_callback(self, msg):
        
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        self.scatter.set_offsets([msg.theta_x, msg.theta_y])
        self.current_x.append(msg.theta_x)
        self.current_y.append(msg.theta_y)
        self.line.set_xdata(self.current_x)
        self.line.set_ydata(self.current_y)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.0001)
        # update the current state

def main(args=None):
    rclpy.init(args=args)

    viz = Visualizer()

    rclpy.spin(viz)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    viz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()