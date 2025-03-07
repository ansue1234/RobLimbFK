import rclpy
from rclpy.node import Node
import torch

from interfaces.msg import Throttle, State, Angles
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ
from matplotlib import pyplot as plt

class Simulator(Node):
    def __init__(self):
        super().__init__('simulator_predictor')
        # Declare parameters (adjust defaults as needed)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('input_size', 6),          # e.g. [t_last, t_now, theta_x, theta_y, throttle_x, throttle_y] if no_time==False and vel==False
                ('seq_len', 100),
                ('hidden_size', 512),
                ('num_layers', 3),
                ('attention', False),
                ('vel', True),             # if True state is [theta_x, theta_y, vel_x, vel_y]
                ('no_time', False),         # if True, time is not included in input
                ('freq', 20),               # update frequency in Hz
                ('throttle_scaling', 10.0),  # scale factor for throttle inputs
                ('dt', 0.075),              # time step in seconds
                ('visualize', True)         # if True, visualize the movement
            ]
        )
        # Retrieve parameters
        self.seq_len = self.get_parameter('seq_len').get_parameter_value().integer_value
        input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        hidden_size = self.get_parameter('hidden_size').get_parameter_value().integer_value
        num_layers = self.get_parameter('num_layers').get_parameter_value().integer_value
        attention = self.get_parameter('attention').get_parameter_value().bool_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.vel = self.get_parameter('vel').get_parameter_value().bool_value
        self.no_time = self.get_parameter('no_time').get_parameter_value().bool_value
        self.freq = self.get_parameter('freq').get_parameter_value().integer_value
        self.throttle_scaling = self.get_parameter('throttle_scaling').get_parameter_value().double_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value

        self.state_pub = self.create_publisher(State, 'state', 1)
        self.angle_pub = self.create_publisher(Angles, 'limb_angles', 1)

        # Subscriber: listen for throttle commands (same as Arduino bridge)
        self.throttle_sub = self.create_subscription(Throttle, 'throttle', self.throttle_callback, 10)
        self.goal_sub = self.create_subscription(Angles, 'goal', self.goal_callback, 1)
        # Timer for periodic simulation updates
        timer_period = 1.0 / self.freq
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize simulation state (position and velocity)
        self.current_state = State()
        self.current_state.theta_x = 0.0
        self.current_state.theta_y = 0.0
        self.current_state.vel_x = 0.0
        self.current_state.vel_y = 0.0


        # Buffer for sequence inputs for the model
        self.data_input = None

        # Latest throttle command (default to zeros)
        self.throttle_queue = []
        self.current_throttle = (0.0, 0.0)

        # Initialize the ML model and its hidden state
        self.hidden = (torch.zeros(num_layers, 1, hidden_size),
                       torch.zeros(num_layers, 1, hidden_size))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FK_SEQ2SEQ(input_size=input_size,
                                    embedding_size=hidden_size,
                                    num_layers=num_layers,
                                    batch_size=1,
                                    output_size=4,
                                    device=self.device,
                                    batch_first=True,
                                    encoder_type='LSTM',
                                    decoder_type='LSTM',
                                    attention=attention,
                                    pred_len=1,
                                    teacher_forcing_ratio=0.0).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.encoder.h0, self.model.encoder.c0 = self.hidden
        self.model.eval()

        # Initialize time management
        self.curr_time = self.get_clock().now()
        self.goal = None
        # code for visualization
        if self.visualize:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.ax.set_xlim(-100, 100)
            self.ax.set_ylim(-100, 100)
            self.current_x, self.current_y = [], []
            self.line_gnd_truth, = self.ax.plot(self.current_x, self.current_y, 'r-', markersize=10, linewidth=4, label='Movement')
            self.scatter = self.ax.scatter([], [], c='purple', s=100, label='Limb Pos.')
            plt.draw()
            plt.pause(0.0001)

    def throttle_callback(self, msg: Throttle):
        # Update the latest throttle command from incoming messages
        thr_x = msg.throttle_x
        thr_y = msg.throttle_y
        if len(self.throttle_queue) != 0:
            self.throttle_queue.pop(0)
        self.throttle_queue.append((thr_x, thr_y))
        # self.current_throttle = (msg.throttle_x, msg.throttle_y)
        self.get_logger().info(f"Received throttle: {msg.throttle_x}, {msg.throttle_y}")

    def _get_state_vector(self):
        # Return current state as a tuple. If velocities are not used, return just angles.
        return (self.current_state.theta_x, self.current_state.theta_y,
                self.current_state.vel_x, self.current_state.vel_y)

    def _prep_input(self):
        # Scale throttle values (similar to fk_predictor)
        thr_scaled = (self.current_throttle[0] * self.throttle_scaling,
                      self.current_throttle[1] * self.throttle_scaling)
        state_vector = self._get_state_vector()
        curr_input = torch.tensor([*state_vector, *thr_scaled],
                                    device=self.device, dtype=torch.float32)
        return curr_input

    def timer_callback(self):
        # Update time stamps
        self.last_time = self.curr_time
        self.curr_time = self.get_clock().now()
        
        if len(self.throttle_queue) != 0:
            self.current_throttle = self.throttle_queue.pop(0)
        else:
            self.current_throttle = (0.0, 0.0)
        # Prepare the current input vector for the model
        curr_input = self._prep_input()
        # Accumulate the input sequence (sliding window of seq_len steps)
        if self.data_input is None:
            self.data_input = curr_input.unsqueeze(0).unsqueeze(0)  # shape [1, input_dim]
        else:
            self.data_input = torch.cat((self.data_input, curr_input.unsqueeze(0).unsqueeze(0)), dim=1)
            if self.data_input.shape[1] > self.seq_len:
                self.data_input = self.data_input[:, -self.seq_len:, :]


        # Run model prediction (compute state delta)
        with torch.no_grad():
            # Add a batch dimension: shape [1, seq_len, input_dim]
            
            input_seq = self.data_input
                # For SEQ2SEQ model
            delta_states, self.hidden = self.model(input_seq, None, self.hidden, mode='test')
            # Assume delta_states is of shape [1, 1, 4]
            delta = delta_states.squeeze(0).squeeze(0)  # tensor of shape [4]

        # Update the predicted state based on the model’s delta output.
        self.current_state.theta_x = self.current_state.theta_x + delta[0].item()
        self.current_state.theta_y = self.current_state.theta_y + delta[1].item()
        self.current_state.vel_x = delta[0].item()/self.dt
        self.current_state.vel_y = delta[1].item()/self.dt

        # Publish the updated states and simulated sensor (angles)
        self._publish_state()
        
        if self.visualize:
            self.current_x.append(self.current_state.theta_x)
            self.current_y.append(self.current_state.theta_y)
            self._viz()

        # Log the results to file and the ROS logger
        log_line = (
            f"State: {self.current_state.theta_x}, {self.current_state.theta_y}, {self.current_state.vel_x}, {self.current_state.vel_y}\n"
            f"Throttle: {self.current_throttle[0]}, {self.current_throttle[1]}\n"
            f"Time: {self.curr_time.nanoseconds * 1e-9}\n"
        )
        self.get_logger().info(log_line)

    def _publish_state(self):
        # Publish sensor message (Angles)
        # Publish full state messages
        ang = Angles()
        ang.theta_x = self.current_state.theta_x
        ang.theta_y = self.current_state.theta_y
        self.angle_pub.publish(ang)
        self.state_pub.publish(self.current_state)
    
    def _viz(self):
        if self.goal:
            self.scatter.set_offsets([self.goal.theta_x, self.goal.theta_y])
        self.line_gnd_truth.set_xdata(self.current_x)
        self.line_gnd_truth.set_ydata(self.current_y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc='upper right', fontsize=18)
        plt.draw()
        plt.pause(0.0001)
    
    def goal_callback(self, msg: Angles):
        self.goal = msg
        
def main(args=None):
    rclpy.init(args=args)
    simulator = Simulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
