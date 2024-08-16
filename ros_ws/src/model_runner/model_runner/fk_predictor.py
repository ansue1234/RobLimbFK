import torch
import rclpy
from rclpy.node import Node

from interfaces.msg import Angles, Throttle, States
from robo_limb_ml.models import fk_lstm

class Predictor(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('FK_predictor')
        self.declare_parameters(
		namespace='/',
		parameters=[
			('model_path', ''), #device we are trasmitting to & receiving messages from
            ('model_type', 'LSTM'),
		]
		)
        # self.com_port = self.get_param_str('device')
        # self.freq = self.get_param_int('frequency')
        # self.baudrate = self.get_param_int('baudrate')
        self.actual_state_publisher_ = self.create_publisher(States, 'actual_state', 1)
        self.pred_publisher_ = self.create_publisher(States, 'pred_state', 1)
        self.act_subscriber_ = self.create_subscription(Throttle, 'throttle', self.throttle_listener_callback, 2)
        self.angle_subscriber_ = self.create_subscription(Angles, 'limb_angles', self.angle_listener_callback, 2)
        
        self.model_path = self.get_param_str('model_path')
        self.model_type = self.get_param_str('model_type')
        # if self.model_type == 'LSTM':
        #     self.model = 

    def get_param_float(self, name):
        try:
            return float(self.get_parameter(name).get_parameter_value().double_value)
        except:
            pass

    def get_param_int(self, name):
        try:
            print(self.get_parameter(name).get_parameter_value().double_value)
            return int(self.get_parameter(name).get_parameter_value().double_value)
        except:
            pass

    def get_param_str(self, name):
        try:
            return self.get_parameter(name).get_parameter_value().string_value
        except:
            pass

    def timer_callback(self):
        # reading sensor and publishing to topic
        try:
            data = self.serial.readline().decode(encoding='utf-8').strip()
            if 'Data' in data:
                self.get_logger().info(data)
                parsed_data = data.split(',')
                msg = Angles()
                msg.theta_x = float(parsed_data[1])
                msg.theta_y = float(parsed_data[2])
                self.publisher_.publish(msg)
            else:
                self.get_logger().info("Read Sensor failed!!!!")
        except Exception as e:
            print(e)
            self.get_logger().info("Read Serial Failed!!!")
        
        # publishing received commands via serial
            # pop the newest command, if no command received, output 0
        if len(self.throttle_queue) != 0:
            thr_x, thr_y = self.throttle_queue.pop(0)
        else:
            thr_x, thr_y = 0.0, 0.0

        try:
            cmd = f"{thr_x},{thr_y}\n"
            self.serial.write(cmd.encode('utf-8')) 
            self.get_logger().info(f"Sending {thr_x}, {thr_y}")
        except Exception as e:
            print(e)
            self.get_logger().info("Send Command Failed!!!")
        
    # stores the newest received command in a queue
    def listener_callback(self, msg):
        thr_x = msg.throttle_x
        thr_y = msg.throttle_y
        if len(self.throttle_queue) != 0:
            self.throttle_queue.pop(0)
        self.throttle_queue.append((thr_x, thr_y))


def main(args=None):
    rclpy.init(args=args)

    bridge = Bridger()

    rclpy.spin(bridge)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bridge.serial.close()
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()