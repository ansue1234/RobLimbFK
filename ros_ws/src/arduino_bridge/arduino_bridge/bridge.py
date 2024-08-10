import rclpy
from rclpy.node import Node
import serial

from interfaces.msg import Angles, Throttle

class Bridger(Node):

    def __init__(self):
        super().__init__('arduino_bridge')
        self.declare_parameters(
		namespace='/',
		parameters=[
			('device', '/dev/ttyACM0'), #device we are trasmitting to & recieving messages from
		    ('baudrate', 9600),
		    ('frequency', 100),
		]
		)
        # self.com_port = self.get_param_str('device')
        # self.freq = self.get_param_int('frequency')
        # self.baudrate = self.get_param_int('baudrate')
        self.com_port = '/dev/ttyACM0'
        self.freq = 100
        self.baudrate = 9600
        self.publisher_ = self.create_publisher(Angles, 'limb_angles', 1)
        timer_period = 1/self.freq # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.serial = serial.Serial(self.com_port, self.baudrate, timeout=0.1)

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
        try:
            data = self.serial.readline().decode(encoding='latin1').strip()
            if 'Data' in data:
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


def main(args=None):
    rclpy.init(args=args)

    bridge = Bridger()

    rclpy.spin(bridge)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()