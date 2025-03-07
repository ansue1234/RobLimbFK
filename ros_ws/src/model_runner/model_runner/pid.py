import rclpy
import numpy as np
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from std_msgs.msg import Bool

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.previous_time = None  # Tracks the time of the previous call

    def compute(self, error, velocity, current_time):
        # Calculate time difference since last update
        if self.previous_time is not None:
            dt = (current_time - self.previous_time).nanoseconds * 1e-9
            self.integral += error * dt
        else:
            dt = 0.0  # No previous time, skip integral update

        # Compute PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * -velocity)
        
        # Update previous time for next iteration
        self.previous_time = current_time
        
        return output

class PIDRunner(Node):
    def __init__(self):
        super().__init__('Policy_runner')
        # Declare PID parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('pid_x.kp', 0.5),
                ('pid_x.ki', 0.0),
                ('pid_x.kd', 0.1),
                ('pid_y.kp', 0.5),
                ('pid_y.ki', 0.0),
                ('pid_y.kd', 0.1),
                ('use_state', True)
            ]
        )

        # Retrieve PID parameters
        kp_x = self.get_parameter('pid_x.kp').get_parameter_value().double_value
        ki_x = self.get_parameter('pid_x.ki').get_parameter_value().double_value
        kd_x = self.get_parameter('pid_x.kd').get_parameter_value().double_value
        kp_y = self.get_parameter('pid_y.kp').get_parameter_value().double_value
        ki_y = self.get_parameter('pid_y.ki').get_parameter_value().double_value
        kd_y = self.get_parameter('pid_y.kd').get_parameter_value().double_value
        
        # Whether to use state subscriber or not
        self.use_state = self.get_parameter('use_state').get_parameter_value().double_value
        # Initialize PID controllers
        self.pid_x = PIDController(kp_x, ki_x, kd_x)
        self.pid_y = PIDController(kp_y, ki_y, kd_y)

        # Setup ROS publishers and subscribers
        self.throttle_publisher_ = self.create_publisher(Throttle, 'throttle', 1)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 1)
        self.goal_subscriber_ = self.create_subscription(Angles, 'goal', self.goal_callback, 1)
        self.control_subscriber_ = self.create_subscription(Bool, 'controller', self.controller_state_callback, 1)
        self.state_subscriber_ = self.create_subscription(State, 'state', self.state_callback, 1)
        
        # State variables
        self.curr_ang = None
        self.past_ang = None
        self.curr_time = None
        self.past_time = None
        self.curr_state = None
        self.goal = Angles()  # Default goal
        self.start = False

    def angle_listener_callback(self, msg):
        # Update past and current angles and time
        self.past_ang = self.curr_ang
        self.past_time = self.curr_time
        
        self.curr_ang = msg
        self.curr_time = self.get_clock().now()

        # Handle first callback
        if self.past_time is None:
            self.past_time = self.curr_time
        if self.past_ang is None:
            self.past_ang = self.curr_ang

        # Calculate current state (including velocities)
        if not self.use_state:
            self.curr_state = State()
            self.curr_state.theta_x = self.curr_ang.theta_x
            self.curr_state.theta_y = self.curr_ang.theta_y
            try:
                dt = (self.curr_time - self.past_time).nanoseconds * 1e-9
                self.curr_state.vel_x = (self.curr_ang.theta_x - self.past_ang.theta_x) / dt
                self.curr_state.vel_y = (self.curr_ang.theta_y - self.past_ang.theta_y) / dt
            except ZeroDivisionError:
                self.curr_state.vel_x = 0.0
                self.curr_state.vel_y = 0.0

        # Run PID control if enabled
        if self.start:
            self.run_policy(self.curr_state, self.goal)

    def run_policy(self, state, goal):
        # Calculate errors
        error_x = goal.theta_x - state.theta_x
        error_y = goal.theta_y - state.theta_y

        # Compute PID outputs
        output_x = self.pid_x.compute(error_x, state.vel_x, self.curr_time)
        output_y = self.pid_y.compute(error_y, state.vel_y, self.curr_time)

        # Clamp outputs and publish
        throttle = Throttle()
        throttle.throttle_x = float(np.clip(output_x, -1.0, 1.0))
        throttle.throttle_y = float(np.clip(output_y, -1.0, 1.0))
        self.throttle_publisher_.publish(throttle)

    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal: {msg.theta_x}, {msg.theta_y}")
        self.goal = msg

    def controller_state_callback(self, msg):
        self.get_logger().info(f"Received controller state: {msg.data}")
        self.start = msg.data
    
    def state_callback(self, msg):
        # self.get_logger().info(f"Received state: {msg.theta_x}, {msg.theta_y}")
        self.curr_state = msg

def main(args=None):
    rclpy.init(args=args)
    runner = PIDRunner()
    rclpy.spin(runner)
    runner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()