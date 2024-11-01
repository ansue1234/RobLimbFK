
import rclpy
from rclpy.node import Node

from interfaces.msg import Angles, Throttle
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd

class Broadcaster(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('plotter')
        # Declare parameters using declare_parameters

        # Retrieve the parameters
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.gnd_truth_publisher_ = self.create_publisher(Angles, 'ground_truth', 1)
        self.prediction_publisher_ = self.create_publisher(Angles, 'prediction', 1)
        self.prediction_file = pd.read_csv('/home/ansue1234/Research/SML/RobLimbFK/robo_limb_ml/results/oct_31/outputs/outputs_FINETUNE_SEQ2SEQ_ATTENTION_b1024_e400_s25000_finetune_final_1730122927_saw_tooth_blue_vid.csv')
        self.counter = 0
        self.init_time = None
        # self.ax.set_xlabel('X Bending Angle ($\theta_x$ degrees)', fontsize=28)
        # self.ax.set_ylabel('Y Bending Angle ($\theta_y$ degrees)', fontsize=28)
        # self.ax.set_title('Live State Visualization', fontsize=34)
        # rosbag2_2024_10_30-19_00_56_sin_purple_video
        # rosbag2_2024_10_30-19_04_05_square_purple_video
        # rosbag2_2024_10_30-19_18_43_saw_tooth_purple_video
        # rosbag2_2024_10_30-20_50_16_sin_blue_video
        # rosbag2_2024_10_30-20_55_21_square_blue
        # rosbag2_2024_10_30-21_00_15_saw_tooth_blue
        
    # stores the newest received angles
    def angle_listener_callback(self, msg):
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        curr_angle.time = msg.time
        if self.init_time is None:
            self.init_time = curr_angle.time
        else:
            time_since_init = curr_angle.time - self.init_time
            first_t = self.prediction_file.iloc[self.counter]['time_begin']
            print(time_since_init/1000, first_t)
            if abs(first_t - time_since_init/1000) < 0.01:
                pred_angle = Angles()
                pred_angle.theta_x = self.prediction_file.iloc[self.counter]['theta_x']
                pred_angle.theta_y = self.prediction_file.iloc[self.counter]['theta_y']
                self.prediction_publisher_.publish(pred_angle)
                self.counter += 1
            if self.counter == len(self.prediction_file):
                self.counter = len(self.prediction_file) - 1
            elif self.counter >0:
                pred_angle = Angles()
                pred_angle.theta_x = self.prediction_file.iloc[self.counter]['theta_x']
                pred_angle.theta_y = self.prediction_file.iloc[self.counter]['theta_y']
                self.prediction_publisher_.publish(pred_angle)
                self.counter += 1
                
        self.gnd_truth_publisher_.publish(curr_angle)
        
    

def main(args=None):
    rclpy.init(args=args)

    viz = Broadcaster()

    rclpy.spin(viz)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    viz.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()