import binascii
import serial
import struct
import sys
import time
import pandas as pd
from datetime import datetime

def initialize_shimmer(com_port):
    ser = serial.Serial(com_port, 115200)
    ser.flushInput()
    print("Port opening, done.")
    return ser

def read_shimmer_data(ser, capture_duration):
    data = []
    start_time = time.time()

    while time.time() - start_time < capture_duration:
        # Send a command to the Shimmer device to request data
        command = 0x72  # Command to read data (this may need to be adjusted based on your Shimmer's protocol)
        ser.write(struct.pack('B', command))
        time.sleep(0.4)

        buf_len = ser.inWaiting()
        if buf_len >= 4:
            ddata = ser.read(4)
            ack, instream, rsp_cmd, status = struct.unpack('4B', ddata)

            self_cmd = (status & 0x04) >> 2
            sensing = (status & 0x02) >> 1
            docked = (status & 0x01)

            # Print for debugging purposes
            print(f"0x{ack:02x}, 0x{instream:02x}, 0x{rsp_cmd:02x}, 0x{status:02x}, self: {self_cmd}, sensing: {sensing}, docked: {docked}")

            # Capture the elapsed time
            elapsed_time = time.time() - start_time

            # Save data
            data.append([elapsed_time, self_cmd, sensing, docked])
        else:
            ddata = ser.read(buf_len)
            print(binascii.hexlify(ddata))
        time.sleep(0.1)

    return data

def save_to_excel(data, file_name):
    df = pd.DataFrame(data, columns=['Elapsed Time', 'Self Cmd', 'Sensing', 'Docked'])
    df.to_excel(file_name, index=False)

def main(com_port, capture_duration, file_name):
    ser = initialize_shimmer(com_port)
    try:
        data = read_shimmer_data(ser, capture_duration)
        save_to_excel(data, file_name)
    finally:
        ser.close()
        print("Port closed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No device specified")
        print("You need to specify the serial port of the device you wish to connect to")
        print("Example:")
        print("   *.py Com12")
        print("or")
        print("   *.py /dev/rfcomm0")
    else:
        com_port = sys.argv[1]
        capture_duration = 80  # Duration in seconds
        file_name = 'prueba_tst.xlsx'
        main(com_port, capture_duration, file_name)
