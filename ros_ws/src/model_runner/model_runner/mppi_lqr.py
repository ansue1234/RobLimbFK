import rclpy
import numpy as np
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, Bool

from ilqr.controller import MPPIBase, FiniteHorizonLQRController
from ilqr.containers import Dynamics, CEMCost, LimbDynamics, LimbDynamicsLQR


class MPPI_LQRRunner(Node):
    def __init__(self):
        super().__init__('MPPI_controller')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('horizon', 10),
                ('lambda', 1.0),
                ('double_shooting', False),
                ('controller', 'MPPI')
            ]
        )
        
        # Retrieve parameters
        self.horizon = self.get_parameter('horizon').get_parameter_value().integer_value
        self.lamb = self.get_parameter('lambda').get_parameter_value().double_value
        self.double_shooting = self.get_parameter('double_shooting').get_parameter_value().bool_value
        self.controller = self.get_parameter('controller').get_parameter_value().string_value
        self.mppi_dynamics = LimbDynamics()
        self.lqr_dynamics = LimbDynamicsLQR()
        self.lqr_controller = None
        self.mppi_controller = None
        self.mppi = False
        self.lqr = False
        
        # Updated cost with final-cost weighting
        Q = np.eye(2) * 10.0
        R = np.eye(2) * 0.5
        self.cost = CEMCost(Q, R)
        
        if self.controller == 'MPPI':
            self.mppi_controller = MPPIBase(
                dynamics=self.mppi_dynamics,
                cost=self.cost,
                horizon=self.horizon,
                x_dim=2,
                u_dim=2)
            self.mppi = True
        if self.controller == 'LQR':
            self.lqr_controller = FiniteHorizonLQRController(
                dynamics=self.lqr_dynamics,
                cost=self.cost,
                horizon=self.horizon)
            self.lqr = True
        if self.controller == 'MPPI_LQR':
            self.mppi_controller = MPPIBase(
                dynamics=self.mppi_dynamics,
                cost=self.cost,
                horizon=self.horizon,
                x_dim=2,
                u_dim=2)
            self.lqr_controller = FiniteHorizonLQRController(
                dynamics=self.lqr_dynamics,
                cost=self.cost,
                horizon=self.horizon)
            self.mppi = True
            self.lqr = True
        
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


    def run_policy(self):
        # No angles or goals yet?
        if self.prev_state is None:
            return
        
        # Convert angles -> torch Tensors
        # x0 = torch.tensor([self.curr_ang.theta_x, self.curr_ang.theta_y], dtype=torch.float32, device=self.device)
        x0 = np.array([self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y])
        xgoal = np.array([self.goal.theta_x, self.goal.theta_y])
        # x0 = torch.tensor([self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y], dtype=torch.float32, device=self.device)
        # xgoal = torch.tensor([self.goal.theta_x, self.goal.theta_y], dtype=torch.float32, device=self.device)

        # Get first-step control from CEM
        self.prev_u = self.best_u
        if self.lqr and not self.mppi:
            self.best_u = self.lqr_controller.iterate(x0, xgoal)
        elif self.mppi and not self.lqr:
            self.best_u = self.mppi_controller.iterate(x0, xgoal)
        elif self.mppi and self.lqr:
            lqr_u = self.lqr_controller.iterate(x0, xgoal)
            self.best_u = self.mppi_controller.iterate(x0, xgoal, u_init=lqr_u)

        if self.mppi and self.double_shooting and self.prev_u is not None:
            curr_state = np.array([self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y])
            # self.best_u = self.cem.double_shooting(curr_state, self.best_u)
            prev_state = np.array([self.prev_state.theta_x, self.prev_state.theta_y, self.prev_state.vel_x, self.prev_state.vel_y])
            # updates err distribution
            self.mppi_controller.calc_diff(curr_state, prev_state, self.prev_u)
            # updates disturbance distribution
            self.best_u = self.mppi_controller.update_u_disturbance_distribution(curr_state)
            
        self.best_u = np.clip(self.best_u, -10.0, 10.0)
        self.get_logger().info(f"Best u: {self.best_u}")
        # best_u is a 2D vector [u_x, u_y]. Clip & publish
        throttle_msg = Throttle()
        throttle_msg.throttle_x = self.best_u[0]/10
        throttle_msg.throttle_y = self.best_u[1]/10
        self.throttle_pub.publish(throttle_msg)
        if self.mppi:
            self.pub_mat(self.mppi_controller.err_cov, self.err_cov_pub)
            self.pub_vec(self.mppi_controller.err_mu, self.err_mean_pub)
            self.pub_mat(self.mppi_controller.u_disturbance_cov, self.u_cov_pub)
            self.pub_vec(self.mppi_controller.u_disturbance_mu, self.u_mean_pub)
            self.pub_mat(self.mppi_controller.cov, self.cem_cov_pub)
            self.pub_vec(self.mppi_controller.mean, self.cem_mean_pub)
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
    node = MPPI_LQRRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
