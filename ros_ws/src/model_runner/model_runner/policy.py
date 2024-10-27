import torch
import torch.nn as nn
import torch.nn.functional as F
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State

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
            "action_scale", torch.tensor([10, 10], dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor([0, 0], dtype=torch.float32)
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
                ('freq', 20),
                ('results_path', ''),
            ]
        )

        # Retrieve the parameters
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.results_path = self.get_parameter('results_path').get_parameter_value().string_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Actor().to(self.device)
        self.model.load_state_dict(torch.load(self.policy_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        date = datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
        result_file = self.results_path + date + 'R'+str(self.threshold) + 'limb_9.txt'
        self.result_file = open(result_file, 'w')

        self.throttle_publisher_ = self.create_publisher(Throttle, 'throttle', 1)
        self.throttle_subscriber_ = self.create_subscription(Throttle, 'raw_throttle', self.throttle_listener_callback, 1)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)

        self.curr_ang = None
        self.past_ang = None
        self.curr_time = None
        self.past_time = None
        self.curr_state = None

        self.current_action = None
        
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
    
    def goal_callback(self, msg):
        self.goal = msg

def main(args=None):
    rclpy.init(args=args)

    filter = PolicyRunner()

    rclpy.spin(filter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()