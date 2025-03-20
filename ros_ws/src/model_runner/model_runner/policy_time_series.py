import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rclpy
from rclpy.node import Node
from interfaces.msg import Angles, Throttle, State
from std_msgs.msg import Bool

# Import gym spaces to create dummy observation and action spaces.
from gymnasium.spaces import Box
# Import the RLAgent from your Actor_Critic_net.py file.
from robo_limb_rl.arch.Actor_Critic_net import RLAgent
from robo_limb_rl.utils.utils import last_items

class PolicyTimeSeriesRunner(Node):
    """
    A ROS node that loads a saved RLAgent (trained with SAC using a time-series input)
    and uses it to compute throttle commands. This version maintains a sliding window
    of the last 100 observations. Each observation is represented as a 6-dimensional
    vector where the first four values correspond to state (theta_x, theta_y, vel_x, vel_y)
    and the last two are zeros (placeholders for the goal). The actual goal (a 2-dimensional
    vector) is provided separately to the agent.
    """
    def __init__(self):
        super().__init__('Policy_runner')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('policy_path', ''),
                ('head_type', 'seq2seq_encoder'),
                ('hidden_dim', 512),
                ('num_layers', 3),
            ]
        )

        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        head_type = self.get_parameter('head_type').get_parameter_value().string_value
        hidden_dim = self.get_parameter('hidden_dim').get_parameter_value().integer_value
        num_layers = self.get_parameter('num_layers').get_parameter_value().integer_value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create dummy observation and action spaces.
        # Observation space here is defined with shape (8,) since during training the agent expected
        # a state vector of 6 numbers (first 6 features) and additional goal features (last 2).
        # In our implementation, we maintain the state history as a sequence of 6-dimensional vectors,
        # and pass the actual goal separately.
        observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        # Assume the action space is 2-dimensional in the range [-1, 1].
        action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Instantiate the RLAgent as defined in Actor_Critic_net.py.
        self.model = RLAgent(observation_space, action_space,
                             head_type=head_type,
                             agent='SAC',
                             hidden_dim=hidden_dim,
                             num_layers=num_layers,
                             batch_size=1,
                             freeze_head=False,
                             pretrained_model=None,
                             device=self.device).to(self.device)

        # Load the saved state dictionary.
        self.model.load_state_dict(torch.load(self.policy_path, map_location=self.device))
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
        self.throttle = Throttle()
        self.throttle.throttle_x = 0.0
        self.throttle.throttle_y = 0.0

        # Buffer to store the last 100 observations (each a 6-dim vector).
        self.obs_buffer = []
        self.start = False
        self.goal = None  # Will be set when a goal message is received
        
    def angle_listener_callback(self, msg):
        # Shift the previous state.
        self.past_ang = self.curr_ang
        self.past_time = self.curr_time
        
        # Unpack the incoming angles message.
        curr_angle = Angles()
        curr_angle.theta_x = msg.theta_x
        curr_angle.theta_y = msg.theta_y
        curr_time = self.get_clock().now()
        
        # Handle first iteration.
        if self.past_time is None:
            self.past_time = curr_time
        if self.past_ang is None:
            self.past_ang = curr_angle
        
        # Build the current state message with angular velocities.
        self.curr_state = State()
        self.curr_state.theta_x = curr_angle.theta_x
        self.curr_state.theta_y = curr_angle.theta_y
        try:
            self.curr_state.vel_x = (curr_angle.theta_x - self.past_ang.theta_x) / ((curr_time - self.past_time).nanoseconds * 1e-9)
            self.curr_state.vel_y = (curr_angle.theta_y - self.past_ang.theta_y) / ((curr_time - self.past_time).nanoseconds * 1e-9)
        except Exception:
            self.curr_state.vel_x = 0.0
            self.curr_state.vel_y = 0.0
            
        self.curr_ang = curr_angle
        self.curr_time = curr_time
        
        if self.goal is not None:
            # Create a 6-dimensional observation vector.
            # The first four values are the state; the last two are placeholders (0.0).
            obs_vec = np.array([self.curr_state.theta_x,
                                self.curr_state.theta_y,
                                self.curr_state.vel_x,
                                self.curr_state.vel_y,
                                self.throttle.throttle_x,
                                self.throttle.throttle_y,
                                self.goal.theta_x,
                                self.goal.theta_y, 0, 0], dtype=np.float32)
            
            # Append the new observation to the buffer and keep only the most recent 100.
            self.obs_buffer.append(obs_vec)
            if len(self.obs_buffer) > 100:
                self.obs_buffer.pop(0)
        
        # If the controller is active, a goal is set, and we have a full window, run the policy.
        if self.start and (self.goal is not None) and len(self.obs_buffer) == 100:
            self.run_policy()
    
    def run_policy(self):
        """
        Uses the last 100 observations as a time-series input and the current goal (from self.goal)
        to compute an action using the loaded RLAgent. The observation input is passed as a tuple:
          - The first element is a tensor of shape [1, 100, 6] representing the state history.
          - The second element is a tensor of shape [1, 2] representing the current goal.
        """
        # Convert the observation buffer into a tensor with shape (1, 100, 6).
        obs_window = torch.tensor(self.obs_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Build the goal tensor (2-dimensional: [goal.theta_x, goal.theta_y]).
        # goal_tensor = torch.tensor([[self.goal.theta_x, self.goal.theta_y]], dtype=torch.float32).to(self.device)
        # Get the action from the RLAgent using tuple input.
        action, _, _ = self.model.get_action(obs_window)
        
        throttle = Throttle()
        thr = action.detach().cpu().numpy().squeeze()
        # Clip the actions to the desired range (adjust scaling if necessary).
        throttle.throttle_x = float(np.clip(thr[0], -1, 1))
        throttle.throttle_y = float(np.clip(thr[1], -1, 1))
        self.throttle_publisher_.publish(throttle)
    
    def goal_callback(self, msg):
        # Save the latest goal message.
        self.goal = msg
    
    def controller_state_callback(self, msg):
        self.start = msg.data
        

def main(args=None):
    rclpy.init(args=args)
    runner = PolicyTimeSeriesRunner()
    rclpy.spin(runner)
    runner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
