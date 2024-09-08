
import pandas as pd
import rclpy
from rclpy.node import Node

from interfaces.msg import Throttle

class OpenLoop(Node):
    # This node receives throttle and then relays the throttle to the arduino
    # all predicted states and actual states are offsetted by 1 timestep
    def __init__(self):
        super().__init__('open_loop_traj')
        # Declare parameters using declare_parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('traj_path', ''),
                ('frequency', 20)
            ]
        )

        # Retrieve the parameters
        self.traj_path = self.get_parameter('traj_path').get_parameter_value().string_value
        self.freq = self.get_parameter('frequency').get_parameter_value().integer_value

        self.thr_publisher_ = self.create_publisher(Throttle, 'throttle', 1)
        
        timer_period = 1/self.freq # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.traj_pts = pd.read_csv(self.traj_path)
        self.counter = 0
    

    def timer_callback(self):
        # read the trajectory file
        # self.counter %= len(self.traj_pts)
        if self.counter < len(self.traj_pts):
            thr_x = self.traj_pts['throttle_x'][self.counter]
            thr_y = self.traj_pts['throttle_y'][self.counter]
            msg = Throttle()
            msg.throttle_x = thr_x
            msg.throttle_y = thr_y
            self.thr_publisher_.publish(msg)
            self.counter += 1
                
def main(args=None):
    rclpy.init(args=args)

    looper = OpenLoop()

    rclpy.spin(looper)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    looper.serial.close()
    looper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()