

import sys
import threading

from interfaces.msg import Throttle
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)



def main():
    settings = saveTerminalSettings()

    rclpy.init()

    node = rclpy.create_node('teleop_twist_keyboard')

    pub = node.create_publisher(Throttle, 'throttle', 10)

    spinner = threading.Thread(target=rclpy.spin, args=(node,))
    spinner.start()

    moveBindings = ['w', 'a', 's', 'd', 'q', 'e', 'z', 'x']

    throttle_msg = Throttle()

    try:
        print("wasdqezx to move")
        throttle_msg = Throttle()
        while True:
            key = getKey(settings)
            if key in moveBindings:
                if key == 'w':
                    throttle_msg.throttle_x = 0.0
                    throttle_msg.throttle_y = 0.5
                elif key == 'a':
                    throttle_msg.throttle_x = -0.5
                    throttle_msg.throttle_y = 0.0
                elif key == 's':
                    throttle_msg.throttle_x = 0.0
                    throttle_msg.throttle_y = -0.5
                elif key == 'd':
                    throttle_msg.throttle_x = 0.5
                    throttle_msg.throttle_y = 0.0
                elif key == 'q':
                    throttle_msg.throttle_x = -0.5
                    throttle_msg.throttle_y = 0.5
                elif key == 'e':
                    throttle_msg.throttle_x = 0.5
                    throttle_msg.throttle_y = 0.5
                elif key == 'z':
                    throttle_msg.throttle_x = -0.5
                    throttle_msg.throttle_y = -0.5
                elif key == 'x':
                    throttle_msg.throttle_x = 0.5
                    throttle_msg.throttle_y = -0.5
            else:
                throttle_msg.throttle_x = 0.0
                throttle_msg.throttle_y = 0.0
                if (key == '\x03'):
                    break
            pub.publish(throttle_msg)


    except Exception as e:
        print(e)
    finally:
        throttle_msg = Throttle()
        throttle_msg.throttle_x = 0.0
        throttle_msg.throttle_y = 0.0
        pub.publish(throttle_msg)
        rclpy.shutdown()
        spinner.join()

        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()