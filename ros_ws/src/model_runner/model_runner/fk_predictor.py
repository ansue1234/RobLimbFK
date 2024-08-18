import torch
import rclpy
import datetime
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, State
from robo_limb_ml.models.fk_lstm import FK_LSTM
from robo_limb_ml.models.fk_seq2seq import FK_SEQ2SEQ

class Predictor(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('FK_predictor')
        # Declare parameters using declare_parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', ''),
                ('input_size', 6),
                ('seq_len', 50),
                ('hidden_size', 512),
                ('num_layers', 3),
                ('attention', False),
                ('model_type', 'LSTM'),
                ('vel', False),
                ('no_time', False),
                ('freq', 20),
                ('rollout', True),
                ('results_path', '')
            ]
        )

        # Retrieve the parameters
        self.seq_len = self.get_parameter('seq_len').get_parameter_value().integer_value
        self.rollout = self.get_parameter('rollout').get_parameter_value().bool_value
        input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        hidden_size = self.get_parameter('hidden_size').get_parameter_value().integer_value
        num_layers = self.get_parameter('num_layers').get_parameter_value().integer_value
        attention = self.get_parameter('attention').get_parameter_value().bool_value
        model_type = self.get_parameter('model_type').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.results_path = self.get_parameter('results_path').get_parameter_value().string_value
        self.vel = self.get_parameter('vel').get_parameter_value().bool_value
        self.no_time = self.get_parameter('no_time').get_parameter_value().bool_value

        self.actual_state_publisher_ = self.create_publisher(State, 'actual_state', 1)
        self.pred_publisher_ = self.create_publisher(State, 'pred_state', 1)
        self.thr_publisher_ = self.create_publisher(Throttle, 'throttle', 1)
        self.act_subscriber_ = self.create_subscription(Throttle, 'raw_throttle', self.throttle_listener_callback, 2)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 2)
        
        self.freq = self.get_parameter('freq').get_parameter_value().integer_value
        timer_period = 1/self.freq # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.throttle_queue = []
        self.curr_ang = None
        self.past_ang = None
        self.curr_time = None
        self.past_time = None
        self.curr_state = None
        self.pred_next_state = None
        self.data_input = None
        self.act_subscriber_
        self.angle_subscriber_
        
        
        # Setting up model
        self.hidden = (torch.zeros(num_layers, 1, hidden_size),
                       torch.zeros(num_layers, 1, hidden_size))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'LSTM':
            self.model = FK_LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_size=1,
                                 output_size=4,
                                 device=self.device,
                                 batch_first=True).to(device=self.device)
            self.model.load_state_dict(torch.load(model_path), map_location=self.device)
            self.model.h0, self.model.c0 = self.hidden
        elif model_type == 'SEQ2SEQ':
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
                                    teacher_forcing_ratio=0.0).to(device=self.device)
            self.model.load_state_dict(torch.load(model_path), map_location=self.device)
            self.model.encoder.h0, self.model.encoder.c0 = self.hidden
        self.model.eval()
    
    def _get_throttle(self):
        if len(self.throttle_queue) != 0:
            return self.throttle_queue[0]
        return (0, 0)
    
    def _get_pred(self):
        input_val = self._prep_input()
        with torch.no_grad():
            if self.model_type == 'LSTM':
                hn, cn = self.hidden
                delta_states, hn, cn = self.model(input_val.unsqueeze(0), hn, cn)
                self.hidden = (hn, cn)
            elif self.model_type == 'SEQ2SEQ':
                delta_states, self.hidden = self.model(input_val.unsqueeze(0), None, self.hidden, mode='test')
        
        return delta_states.squeeze(0)
    
    def _get_state(self):
        if not self.rollout:
            if not self.vel:
                return self.curr_state.theta_x, self.curr_state.theta_y
            return self.curr_state.theta_x, self.curr_state.theta_y, self.curr_state.vel_x, self.curr_state.vel_y
        else:
            if not self.vel:
                return self.pred_next_state.theta_x, self.pred_next_state.theta_y
            return self.pred_next_state.theta_x, self.pred_next_state.theta_y, self.pred_next_state.vel_x, self.pred_next_state.vel_y
            
    def _prep_input(self):
        if self.no_time:
            curr_input = torch.tensor([*self._get_state(), *self._get_throttle()]).to(device=self.device)
        else:
            curr_input = torch.tensor([self.curr_time.nanoseconds*1e-9, self.self.curr_time.nanoseconds*1e-9, *self._get_state(), *self._get_throttle()]).to(device=self.device)
        
        # stacking to seq_len
        if self.data_input is None:
            self.data_input = curr_input.unsqueeze(0)
        else:
            self.data_input = torch.cat((self.data_input, curr_input.unsqueeze(0)), dim=0)
        if len(self.data_input) > self.seq_len:
            self.data_input = self.data_input[1:]
        return self.data_input
        
    
    # stores the newest received command in a queue
    def throttle_listener_callback(self, msg):
        thr_x = msg.throttle_x
        thr_y = msg.throttle_y
        if len(self.throttle_queue) != 0:
            self.throttle_queue.pop(0)
        self.throttle_queue.append((thr_x, thr_y))

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
            self.past_ang = self.curr_ang
        
        # update the current state
        self.curr_state = State()
        self.curr_state.theta_x = curr_angle.theta_x
        self.curr_state.theta_y = curr_angle.theta_y
        self.curr_state.vel_x = (curr_angle.theta_x - self.past_ang.theta_x) / ((curr_time - self.past_time).nanoseconds*1e-9)
        self.curr_state.vel_y = (curr_angle.theta_y - self.past_ang.theta_y) / ((curr_time - self.past_time).nanoseconds*1e-9)
        
        self.curr_ang = curr_angle
        self.curr_time = curr_time
    
    def timer_callback(self):
        if self.curr_state is not None:
            delta_states = self._get_pred()
            
            self.actual_state_publisher_.publish(self.curr_state)
            self.pred_publisher_.publish(self.pred_next_state)
            
            # append file pathe with today's date
            date = datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
            result_file = self.results_path + date + '.txt'
            with open(result_file, 'w') as f:
                f.write(f"Actual, {self.curr_state.theta_x}, {self.curr_state.theta_y}, {self.curr_state.vel_x}, {self.curr_state.vel_y}\n")
                f.write(f"Predicted, {self.pred_next_state.theta_x}, {self.pred_next_state.theta_y}, {self.pred_next_state.vel_x}, {self.pred_next_state.vel_y}\n")
                
            if self.rollout:
                self.pred_next_state.theta_x += delta_states[0]
                self.pred_next_state.theta_y += delta_states[1]
                self.pred_next_state.vel_x += delta_states[2]
                self.pred_next_state.vel_y += delta_states[3]
        

def main(args=None):
    rclpy.init(args=args)

    predictor = Predictor()

    rclpy.spin(predictor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    predictor.serial.close()
    predictor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()