import torch
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles
from robo_limb_rl.arch.Q_net import QNet_MLP
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rc('font',family='Times New Roman')

class Visualizer(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('visualizer')
        # Declare parameters using declare_parameters

        # Retrieve the parameters
        self.angle_subscriber_ = self.create_subscription(Angles, 'ground_truth', self.angle_listener_callback, 1)
        self.prediction_subscriber_ = self.create_subscription(Angles, 'prediction', self.prediction_listener_callback, 1)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        # self.ax.set_xlabel('X Bending Angle ($\theta_x$ degrees)', fontsize=28)
        # self.ax.set_ylabel('Y Bending Angle ($\theta_y$ degrees)', fontsize=28)
        # self.ax.set_title('Live State Visualization', fontsize=34)
        self.current_x, self.current_y = [], []
        self.predicted_x, self.predicted_y = [], []
        self.line_gnd_truth, = self.ax.plot(self.current_x, self.current_y, 'r-', markersize=10, linewidth=4, label='Gnd. Truth')
        self.line_pred, = self.ax.plot(self.predicted_x, self.predicted_y, color='blue', linestyle='dotted', markersize=10, linewidth=4, label='Pred. (Fine-tuned)')
        # self.scatter = self.ax.scatter([], [], c='purple', s=100, label='Limb Pos.')
        # self.ax.plot(30*np.cos(np.linspace(0, 2*np.pi, 100)), 30*np.sin(np.linspace(0, 2*np.pi, 100)), 'g', label='Safe Bound.', linewidth=3)
        plt.draw()
        plt.pause(0.0001)
        
    # stores the newest received angles
    def angle_listener_callback(self, msg):
        
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        self.current_x.append(msg.theta_x)
        self.current_y.append(msg.theta_y)
    
    
    def prediction_listener_callback(self, msg):
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        self.predicted_x.append(msg.theta_x)
        self.predicted_y.append(msg.theta_y)
    
    def timer_callback(self):
        self.line_gnd_truth.set_xdata(self.current_x)
        self.line_gnd_truth.set_ydata(self.current_y)
        self.line_pred.set_xdata(self.predicted_x)
        self.line_pred.set_ydata(self.predicted_y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc='upper right', fontsize=18)
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