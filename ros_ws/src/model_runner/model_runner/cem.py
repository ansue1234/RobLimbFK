import rclpy
import numpy as np
import torch
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Bool

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
                ('cem.horizon', 5),
                ('cem.num_samples', 1000),
                ('cem.num_elite', 100),
                ('cem.max_iters', 1),
                ('cem.alpha', 0.5),
                ('model_path', ''),
                ('double_shooting', True)
            ]
        )
        
        # Retrieve parameters
        horizon = self.get_parameter('cem.horizon').get_parameter_value().integer_value
        num_samples = self.get_parameter('cem.num_samples').get_parameter_value().integer_value
        num_elite = self.get_parameter('cem.num_elite').get_parameter_value().integer_value
        max_iters = self.get_parameter('cem.max_iters').get_parameter_value().integer_value
        alpha = self.get_parameter('cem.alpha').get_parameter_value().double_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.double_shooting = self.get_parameter('double_shooting').get_parameter_value().bool_value
        
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
        self.err_cov_pub = self.create_publisher(Float64MultiArray, 'err_cov', 1)
        self.err_mean_pub = self.create_publisher(Float64MultiArray, 'err_mean', 1)
        self.u_cov_pub = self.create_publisher(Float64MultiArray, 'u_cov', 1)
        self.u_mean_pub = self.create_publisher(Float64MultiArray, 'u_mean', 1)
        self.cem_cov_pub = self.create_publisher(Float64MultiArray, 'cem_cov', 1)
        self.cem_mean_pub = self.create_publisher(Float64MultiArray, 'cem_mean', 1)
        # self.angles_sub = self.create_subscription(Angles, 'limb_angles', self.angles_cb, 1)
        self.state_sub = self.create_subscription(State, 'state', self.state_cb, 1)
        self.goal_sub = self.create_subscription(Angles, 'goal', self.goal_cb, 1)
        self.controller_sub = self.create_subscription(Bool, 'controller', self.controller_state_cb, 1)
        
        
        # State & control toggles
        self.curr_state = None
        self.prev_state = None
        self.goal = Angles()
        self.prev_u = None
        self.best_u = None
        self.start_control = False

    def state_cb(self, msg: State):
        self.prev_state = self.curr_state
        self.curr_state = msg
        # If controller is active, run
        if self.start_control:
            self.run_policy()
    
    def angles_cb(self, msg: Angles):
        self.curr_ang = msg
        # If controller is active, run
        if self.start_control:
            self.run_policy()

    def run_policy(self):
        # No angles or goals yet?
        if self.prev_state is None:
            return
        
        # Convert angles -> torch Tensors
        # x0 = torch.tensor([self.curr_ang.theta_x, self.curr_ang.theta_y], dtype=torch.float32, device=self.device)
        x0 = torch.tensor([self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y], dtype=torch.float32, device=self.device)
        xgoal = torch.tensor([self.goal.theta_x, self.goal.theta_y], dtype=torch.float32, device=self.device)

        # Get first-step control from CEM
        self.prev_u = self.best_u
        self.best_u = self.cem.iterate(x0, xgoal)
        if self.double_shooting and self.prev_u is not None:
            curr_state = np.array([self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y])
            # self.best_u = self.cem.double_shooting(curr_state, self.best_u)
            prev_state = np.array([self.prev_state.theta_x, self.prev_state.theta_y, self.prev_state.vel_x, self.prev_state.vel_y])
            # updates err distribution
            self.cem.calc_diff(curr_state, prev_state, self.prev_u)
            # updates disturbance distribution
            self.best_u = self.cem.update_u_disturbance_distribution(curr_state)
        self.get_logger().info(f"Best u: {self.best_u}")
        # best_u is a 2D vector [u_x, u_y]. Clip & publish
        throttle_msg = Throttle()
        throttle_msg.throttle_x = float(np.clip(self.best_u[0], -1.0, 1.0))
        throttle_msg.throttle_y = float(np.clip(self.best_u[1], -1.0, 1.0))
        self.throttle_pub.publish(throttle_msg)
        self.pub_mat(self.cem.err_cov, self.err_cov_pub)
        self.pub_vec(self.cem.err_mu, self.err_mean_pub)
        self.pub_mat(self.cem.u_disturbance_cov, self.u_cov_pub)
        self.pub_vec(self.cem.u_disturbance_mu, self.u_mean_pub)
        self.pub_mat(self.cem.cov, self.cem_cov_pub)
        self.pub_vec(self.cem.mean, self.cem_mean_pub)
        # self.get_logger().info("Error Cov:")

    def goal_cb(self, msg: Angles):
        # self.get_logger().info(f"Received goal: {msg.theta_x}, {msg.theta_y}")
        self.goal = msg

    def controller_state_cb(self, msg: Bool):
        # self.get_logger().info(f"Controller state: {msg.data}")
        self.start_control = msg.data

    def pub_mat(self, matrix, publisher):
        # Prepare the MultiArray message
        msg = Float64MultiArray()

        # Define layout dimensions: first dimension is rows, second is columns.
        dim0 = MultiArrayDimension()
        dim0.label = "rows"
        dim0.size = matrix.shape[0]
        # Stride for rows is the total number of elements
        dim0.stride = matrix.size

        dim1 = MultiArrayDimension()
        dim1.label = "cols"
        dim1.size = matrix.shape[1]
        # Stride for columns is the number of columns (since they are contiguous in row-major order)
        dim1.stride = matrix.shape[1]

        msg.layout.dim = [dim0, dim1]
        msg.layout.data_offset = 0

        # Flatten the matrix into a list (row-major order)
        msg.data = matrix.flatten().tolist()

        # Publish the message
        publisher.publish(msg)
    
    def pub_vec(self, arr, publisher):
        msg = Float64MultiArray()
        
        # Define a single dimension layout
        dim = MultiArrayDimension()
        dim.label = "elements"
        dim.size = arr.size
        dim.stride = arr.size  # For a 1D array, stride is just the total size.
        msg.layout.dim = [dim]
        msg.layout.data_offset = 0

        # Flatten the array (though it's already 1D) and convert to list
        msg.data = arr.flatten().tolist()

        publisher.publish(msg)
def main(args=None):
    rclpy.init(args=args)
    node = CEMRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
