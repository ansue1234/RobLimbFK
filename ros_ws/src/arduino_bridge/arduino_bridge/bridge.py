import rclpy
from rclpy.node import Node
import serial

from interfaces.msg import Angles, Throttle

class Bridger(Node):

    def __init__(self):
        super().__init__('arduino_bridge')
        self.declare_parameters(
		namespace='',
		parameters=[
			('device', '/dev/ttyACM0'), #device we are trasmitting to & recieving messages from
		    ('baudrate', 9600),
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
        self.subscriber_ = self.create_subscription(Throttle, 'throttle', self.listener_callback, 2)
        
        timer_period = 1/self.freq # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.serial = serial.Serial(self.com_port, self.baudrate, timeout=0.1)
        self.throttle_queue = []
        self.subscriber_


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