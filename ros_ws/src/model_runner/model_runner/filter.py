import torch
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from robo_limb_rl.arch.Q_net import QNet_MLP

class PolicyFilter(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('Policy_filter')
        # Declare parameters using declare_parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('policy_path', ''),
                ('freq', 20),
                ('results_path', ''),
                ('threshold', 90)
            ]
        )

        # Retrieve the parameters
        self.policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.results_path = self.get_parameter('results_path').get_parameter_value().string_value
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = self.get_parameter('threshold').get_parameter_value().integer_value
        self.model = QNet_MLP(input_dim=4, output_dim=4, reward_type='reg').to(self.device)
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

    def _throttle_to_action(self, throttle):
        # throttle is a tuple of floats
        thr_x, thr_y = throttle
        thr_x, thr_y = int(thr_x), int(thr_y)
        if thr_x == 1 and thr_y == 1:
            action = 3
        elif thr_x == 1 and thr_y == -1:
            action = 2
        elif thr_x == -1 and thr_y == 1:
            action = 1
        elif thr_x == -1 and thr_y == -1:
            action = 0
        return action
        
    def _action_to_throttle(self, action):
        # action is an int
        if action == 3:
            return (1.0, 1.0)
        elif action == 2:
            return (1.0, -1.0)
        elif action == 1:
            return (-1.0, 1.0)
        elif action == 0:
            return (-1.0, -1.0)
    
    # stores the newest received command in a queue
    def throttle_listener_callback(self, msg):
        # unpacking nominal throttle
        thr_x = int(msg.throttle_x)
        thr_y = int(msg.throttle_y)
        
        # convert throttle to action
        action = self._throttle_to_action((thr_x, thr_y))
        
        # prepare states as inputs
        if self.curr_state is not None:
            x, y, x_dot, y_dot = self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y
            nom_act = True
            obs = torch.tensor([x, y, x_dot, y_dot], dtype=torch.float32).to(self.device)
            q_vals = self.model(obs)
            if q_vals[action] < self.threshold:
                action = torch.argmax(q_vals)
                nom_act = False
            
            throttle = self._action_to_throttle(action)
            thr_msg = Throttle()
            thr_msg.throttle_x = throttle[0]
            thr_msg.throttle_y = throttle[1]
            
            self.get_logger().info(f"Current state: {x}, {y}, {x_dot}, {y_dot}")
            self.get_logger().info(f"Q values: {q_vals}")
            if nom_act:
                self.get_logger().info(f"nominal action, Throttle: {throttle[0]}, {throttle[1]}, action: {action}, ")
            else:
                self.get_logger().info(f"safe action, Throttle: {throttle[0]}, {throttle[1]}, action: {action}")
            self.get_logger().info("---")
            self.throttle_publisher_.publish(thr_msg)

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
            
        self.result_file.write(f"Actual, {self.curr_state.theta_x}, {self.curr_state.theta_y}, {self.curr_state.vel_x}, {self.curr_state.vel_y}\n")
        self.curr_ang = curr_angle
        self.curr_time = curr_time
    

def main(args=None):
    rclpy.init(args=args)

    filter = PolicyFilter()

    rclpy.spin(filter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()