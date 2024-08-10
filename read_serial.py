import serial
# import matplotlib.pyplot as plt
# import numpy as np

def readserial(comport, baudrate):

    # fig, axs = plt.figure(1, 2, figsize=(10, 5))
    # t = np.array([])
    # px = np.array([])
    # py = np.array([])
    # nx = np.array([])
    # ny = np.array([])
    # line_px, = axs[0].plot(t, px, label='PX')
    # line_nx, = axs[0].plot(t, py, label='NX')
    # line_py, = axs[1].plot(t, nx, label='PY')
    # line_ny, = axs[1].plot(t, ny, label='NY')
    # axs[0].set_xlabel('X')
    # axs[0].set_ylabel('PWM')
    # axs[1].set_xlabel('Y')
    # axs[1].set_ylabel('PWM')

    ser = serial.Serial(comport, baudrate, timeout=0.1)         # 1/timeout is the frequency at which the port is read
    start = False
    while True:
        data = ser.readline().decode(encoding='utf-8').strip()
        # print('Data' in data)
        # try: 
        #     x = data.split(',')[1]
        #     y = data.split(',')[2]
        #     print('x:', float(x), 'y:', float(y))
            
        # except:
        #     pass
        print(data)
        ser.write("Hi\n".encode('utf-8'))
        # if data:
        #     if "Complete" in data:
        #         start = True
        #     if start and "exceeding" not in data:
        #         print(data)
        #         data = data.split(',')
        #         time, theta_x, theta_y, pwm_px, pwm_nx, pwm_py, pwm_ny = [float(i) for i in data]
        #         t = np.append(t, time)
        #         px = np.append(px, pwm_px)
        #         nx = np.append(nx, pwm_nx)
        #         py = np.append(py, pwm_py)
        #         ny = np.append(ny, pwm_ny)
        #         line_px.set_data(t, pwm_px)
        #         line_nx.set_data(t, pwm_nx)
        #         line_py.set_data(t, pwm_py)
        #         line_ny.set_data(t, pwm_ny)

        #         axs[0].draw_artist(axs[0].patch)
        #         axs[1].draw_artist(axs[1].patch)
        #         axs[0].draw_artist(line_px)
        #         axs[0].draw_artist(line_nx)
        #         axs[1].draw_artist(line_py)
        #         axs[1].draw_artist(line_ny)

        #         axs[0].legend()
        #         axs[1].legend()
        #         # ax.draw_artist(tips)
        #         fig.canvas.draw_idle()
        #         fig.show()
        #         fig.canvas.flush_events()


if __name__ == '__main__':

    readserial('/dev/ttyACM0', 9600)
# Current limb -10, -12 deg bias