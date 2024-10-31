
import rclpy
from rclpy.node import Node

from interfaces.msg import Angles, Throttle
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os

class Plotter(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('plotter')
        # Declare parameters using declare_parameters

        # Retrieve the parameters
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.throttle_subscriber_ = self.create_subscription(Throttle, 'sent_throttle', self.throttle_listener_callback, 1)
        self.file = open(os.path.join(os.getcwd(), 'sin_blue_vid_data.txt'), 'a')
        self.file.write('time,theta_x,theta_y\n')
        self.file_throttle = open(os.path.join(os.getcwd(), 'sin_blue_vid_throttle.txt'), 'a')
        self.file_throttle.write('throttle_x,throttle_y\n')
        # self.ax.set_xlabel('X Bending Angle ($\theta_x$ degrees)', fontsize=28)
        # self.ax.set_ylabel('Y Bending Angle ($\theta_y$ degrees)', fontsize=28)
        # self.ax.set_title('Live State Visualization', fontsize=34)
        
    # stores the newest received angles
    def angle_listener_callback(self, msg):
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        curr_angle.time = msg.time
        self.file.write(f'{curr_angle.time},{curr_angle.theta_x},{curr_angle.theta_y}\n')
    
    def throttle_listener_callback(self, msg):
        # unpack the message
        curr_thr = Throttle()
        curr_thr.throttle_x = msg.throttle_x
        curr_thr.throttle_y = msg.throttle_y
        self.file_throttle.write(f'{curr_thr.throttle_x},{curr_thr.throttle_y}\n')
        # update the current state

def main(args=None):
    rclpy.init(args=args)

    viz = Plotter()

    rclpy.spin(viz)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    viz.file.close()
    viz.file_throttle.close()
    viz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()