import serial


def readserial(comport, baudrate):

    ser = serial.Serial(comport, baudrate, timeout=0.1)         # 1/timeout is the frequency at which the port is read

    while True:
        data = ser.readline().decode(encoding='latin1').strip()
        if data:
            print(data)


if __name__ == '__main__':

    readserial('/dev/ttyACM0', 9600)
# Current limb -10, -12 deg bias