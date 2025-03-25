import rclpy
from rclpy.node import Node
import serial

from interfaces.msg import Angles, Throttle,State

class Bridger(Node):

    def __init__(self):
        super().__init__('arduino_bridge')
        self.declare_parameters(
		namespace='',
		parameters=[
			('device', '/dev/ttyACM0'), #device we are trasmitting to & recieving messages from
		    ('baudrate', 115200),
		    ('frequency', 100),
		]
		)
        # self.com_port = self.get_param_str('device')
        # self.freq = self.get_param_int('frequency')
        # self.baudrate = self.get_param_int('baudrate')
        self.com_port = self.get_parameter('device').get_parameter_value().string_value
        self.freq = self.get_parameter('frequency').get_parameter_value().integer_value 
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.publisher_ = self.create_publisher(Angles, 'limb_angles', 1)
        self.publisher_thr = self.create_publisher(Throttle, 'sent_throttle', 1)
        self.state_publisher = self.create_publisher(State, 'state', 1)
        self.subscriber_ = self.create_subscription(Throttle, 'throttle', self.listener_callback, 10)
        self.prev_angle = None
        self.curr_angle = None
        
        timer_period = 1/self.freq # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.serial = serial.Serial(self.com_port, self.baudrate, timeout=1)
        self.throttle_queue = []
        self.subscriber_


    def timer_callback(self):
        # reading sensor and publishing to topic
        try:
            data = self.serial.readline().decode(encoding='utf-8').strip()
            # if data:
            # self.get_logger().info(data)
            #     print(data)
            if 'Data' in data:
                self.get_logger().info(data)
                parsed_data = data.split(',')
                msg = Angles()
                msg.theta_x = float(parsed_data[1])
                msg.theta_y = float(parsed_data[2])
                msg.time = float(parsed_data[5])
                msg.power_px = float(parsed_data[6])
                msg.power_py = float(parsed_data[7])
                msg.power_nx = float(parsed_data[8])
                msg.power_ny = float(parsed_data[9])
                # Doing state calculations
                self.prev_angle = self.curr_angle
                self.curr_angle = msg
                if self.prev_angle is not None and self.curr_angle is not None:
                    state = State()
                    state.theta_x = self.curr_angle.theta_x
                    state.theta_y = self.curr_angle.theta_y
                    state.vel_x = (self.curr_angle.theta_x - self.prev_angle.theta_x)/((self.curr_angle.time - self.prev_angle.time)/1000)
                    state.vel_y = (self.curr_angle.theta_y - self.prev_angle.theta_y)/((self.curr_angle.time - self.prev_angle.time)/1000)
                    self.state_publisher.publish(state)
                thr_msg = Throttle()
                thr_msg.throttle_x = float(parsed_data[3])
                thr_msg.throttle_y = float(parsed_data[4])
                self.publisher_.publish(msg)
                self.publisher_thr.publish(thr_msg)
            else:
                self.get_logger().info("Read Sensor failed!!!! Data: " + data)
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
        if len(self.throttle_queue) >= 10:
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