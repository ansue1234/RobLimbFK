
import pandas as pd
import numpy as np
import rclpy
from rclpy.node import Node

from interfaces.msg import Angles
from std_msgs.msg import Bool

class Feeder(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('traj_feeder')
        # Declare parameters using declare_parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('traj_path', ''),
                ('random', True)
            ]
        )

        # Retrieve the parameters
        self.traj_path = self.get_parameter('traj_path').get_parameter_value().string_value
        self.random = self.get_parameter('random').get_parameter_value().bool_value
        # Create the publishers and subscribers
        self.goal_publisher_ = self.create_publisher(Angles, 'goal', 1)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.controller_publisher_ = self.create_publisher(Bool, 'controller', 1)
        self.traj_pts = pd.read_csv(self.traj_path)
        self.counter = 0
        self.current_goal = np.array([self.traj_pts['goal_x'][self.counter], self.traj_pts['goal_y'][self.counter]])
        self.start = Bool()
        self.start.data = False
    
    def angle_listener_callback(self, msg):
        # gets current angle and compare whether goal is reached
        curr_angle = np.array([msg.theta_x, msg.theta_y])
        if self.start.data is False:
            start = Bool()
            start.data = True
            self.controller_publisher_.publish(start)
            self.start = start
            if self.random:
                self.current_goal = np.array([np.random.uniform(-60, 60), np.random.uniform(-60, 60)])
        # if goal is reached, increment counter/get counter logic
        # find index of the the goal that is closest to the current angle
        pts = self.traj_pts.values
        dist = np.linalg.norm(pts - curr_angle, axis=1)
        counter = np.argmin(dist)
        # if self.counter < counter and np.linalg.norm(curr_angle - self.current_goal) < 10:
        #     self.counter = counter
        if np.linalg.norm(curr_angle - self.current_goal) < 10:
            self.counter += 1
            if self.random:
                self.current_goal = np.array([np.random.uniform(-60, 60), np.random.uniform(-60, 60)])
        
        # send the next goal
        if not self.random:
            if self.counter < len(self.traj_pts):
                goal_x = self.traj_pts['goal_x'][self.counter]
                goal_y = self.traj_pts['goal_y'][self.counter]
                msg = Angles()
                msg.theta_x = goal_x
                msg.theta_y = goal_y
                self.goal_publisher_.publish(msg)
                self.controller_publisher_.publish(self.start)
                self.current_goal = np.array([goal_x, goal_y])
            else:
                # stop if trajectory is finished
                self.start.data = False
                self.controller_publisher_.publish(self.start)
        else:
            msg = Angles()
            msg.theta_x = self.current_goal[0]
            msg.theta_y = self.current_goal[1]
            self.goal_publisher_.publish(msg)
            self.controller_publisher_.publish(self.start)
            # self.current_goal = np.array([goal_x, goal_y])
                
def main(args=None):
    rclpy.init(args=args)

    feeder = Feeder()

    rclpy.spin(feeder)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    feeder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()