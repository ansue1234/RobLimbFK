import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from std_msgs.msg import Bool

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, 2)
        self.fc_logstd = nn.Linear(256, 2)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor([[10, 10]], dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor([[0, 0]], dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class PolicyRunner(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('Policy_runner')
        # Declare parameters using declare_parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('policy_path', ''),
            ]
        )

        # Retrieve the parameters
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Actor().to(self.device)
        print("Hi------------------")
        # print(torch.load(self.policy_path, map_location=self.device, weights_only=True))
        self.model.load_state_dict(torch.load(self.policy_path, map_location=self.device, weights_only=True)[0])
        self.model.eval()

        self.throttle_publisher_ = self.create_publisher(Throttle, 'throttle', 1)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.goal_subscriber_ = self.create_subscription(Angles, 'goal', self.goal_callback, 1)
        self.control_subscriber_ = self.create_subscription(Bool, 'controller', self.controller_state_callback, 1)
        
        self.curr_ang = None
        self.past_ang = None
        self.curr_time = None
        self.past_time = None
        self.curr_state = None

        self.current_action = None
        
        self.start = False
        
    # stores the newest received angles
    def angle_listener_callback(self, msg):
        # shift the current state to the past state
        self.past_ang = self.curr_ang
        self.past_time = self.curr_time
        
        # unpack the message
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        curr_time = self.get_clock().now()
        
        # handle first iteration
        if self.past_time is None:
            self.past_time = curr_time
        if self.past_ang is None:
            self.past_ang = curr_angle
        
        # update the current state
        self.curr_state = State()
        self.curr_state.theta_x = curr_angle.theta_x
        self.curr_state.theta_y = curr_angle.theta_y
        try:
            self.curr_state.vel_x = (curr_angle.theta_x - self.past_ang.theta_x) / ((curr_time - self.past_time).nanoseconds*1e-9)
            self.curr_state.vel_y = (curr_angle.theta_y - self.past_ang.theta_y) / ((curr_time - self.past_time).nanoseconds*1e-9)
        except:
            self.curr_state.vel_x = 0.0
            self.curr_state.vel_y = 0.0
            
        self.curr_ang = curr_angle
        self.curr_time = curr_time
        # Running policy to get throttle
        if self.start:
            self.run_policy(self.curr_state, self.goal)
    
    def run_policy(self, state, goal):
        # state = torch.tensor([state.theta_x, state.theta_y, state.vel_x, state.vel_y], dtype=torch.float32).unsqueeze(0).to(self.device)
        state = torch.tensor([state.theta_x, state.theta_y, state.vel_x, state.vel_y, goal.theta_x, goal.theta_y], dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _, _ = self.model.get_action(state.unsqueeze(0))
        throttle = Throttle()
        thr = action.detach().cpu().numpy().squeeze()
        # self.get_logger().info(f"thr: {thr}")
        throttle.throttle_x = float(np.clip(thr[0]/10, -1, 1))
        throttle.throttle_y = float(np.clip(thr[1]/10, -1, 1))
        self.throttle_publisher_.publish(throttle)
    
    def goal_callback(self, msg):
        self.goal = msg
    
    def controller_state_callback(self, msg):
        self.start = msg.data
        

def main(args=None):
    rclpy.init(args=args)

    runner = PolicyRunner()

    rclpy.spin(runner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    runner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()