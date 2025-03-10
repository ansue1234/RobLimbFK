import rclpy
import numpy as np
import torch
from rclpy.node import Node

from interfaces.msg import Angles, Throttle
from std_msgs.msg import Bool

from ilqr.controller import CEMPlanner
from ilqr.containers import Dynamics, CEMCost
from ilqr.arch import LimbModel


class CEMRunner(Node):
    def __init__(self):
        super().__init__('CEM_controller')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('cem.horizon', 10),
                ('cem.num_samples', 1000),
                ('cem.num_elite', 100),
                ('cem.max_iters', 10),
                ('cem.alpha', 0.5),
                ('model_path', '')
            ]
        )
        
        # Retrieve parameters
        horizon = self.get_parameter('cem.horizon').get_parameter_value().integer_value
        num_samples = self.get_parameter('cem.num_samples').get_parameter_value().integer_value
        num_elite = self.get_parameter('cem.num_elite').get_parameter_value().integer_value
        max_iters = self.get_parameter('cem.max_iters').get_parameter_value().integer_value
        alpha = self.get_parameter('cem.alpha').get_parameter_value().double_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        # Load torch-based dynamics model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The LimbModel path, hidden sizes, etc. can be updated if arch.py changed
        self.model = LimbModel(
            model_path,   # Update to your new path if needed
            input_size=6,       # Adjust if your new arch expects a different input size
            hidden_size=512,
            num_layers=3,
            attention=False,
            device=self.device
        )
        self.dynamics = Dynamics.Torch(self.model, self)
        
        # Updated cost with final-cost weighting
        Q = np.eye(2) * 10.0
        R = np.eye(2) * 0.5
        self.cost = CEMCost(Q, R)
        
        # Build the CEMPlanner
        self.cem = CEMPlanner(
            dynamics=self.dynamics,
            cost=self.cost,
            horizon=horizon,
            x_dim=2,
            u_dim=2,
            num_samples=num_samples,
            num_elite=num_elite,
            max_iters=max_iters,
            alpha=alpha,
            debugger=self
        )
        
        # ROS publishers & subscribers
        self.throttle_pub = self.create_publisher(Throttle, 'throttle', 1)
        self.angles_sub = self.create_subscription(Angles, 'limb_angles', self.angles_cb, 1)
        self.goal_sub = self.create_subscription(Angles, 'goal', self.goal_cb, 1)
        self.controller_sub = self.create_subscription(Bool, 'controller', self.controller_state_cb, 1)
        
        # State & control toggles
        self.curr_ang = None
        self.goal = Angles()
        self.start_control = False

    def angles_cb(self, msg: Angles):
        self.curr_ang = msg
        # If controller is active, run
        if self.start_control:
            self.run_policy()

    def run_policy(self):
        # No angles or goals yet?
        if self.curr_ang is None:
            return
        
        # Convert angles -> torch Tensors
        x0 = torch.tensor([self.curr_ang.theta_x, self.curr_ang.theta_y], dtype=torch.float32, device=self.device)
        xgoal = torch.tensor([self.goal.theta_x, self.goal.theta_y], dtype=torch.float32, device=self.device)

        # Get first-step control from CEM
        best_u = self.cem.iterate(x0, xgoal)
        
        # best_u is a 2D vector [u_x, u_y]. Clip & publish
        throttle_msg = Throttle()
        throttle_msg.throttle_x = float(np.clip(best_u[0], -1.0, 1.0))
        throttle_msg.throttle_y = float(np.clip(best_u[1], -1.0, 1.0))
        self.throttle_pub.publish(throttle_msg)

    def goal_cb(self, msg: Angles):
        self.get_logger().info(f"Received goal: {msg.theta_x}, {msg.theta_y}")
        self.goal = msg

    def controller_state_cb(self, msg: Bool):
        self.get_logger().info(f"Controller state: {msg.data}")
        self.start_control = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = CEMRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
