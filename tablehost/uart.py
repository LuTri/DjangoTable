import serial
import time

import socket
import struct

from django.conf import settings

MAX_LEDS = 336

class UartCom(object):
    def __init__(self, debug=False, baud=500000):
        self.baud = baud
        self.data = []
        self.debug = debug
        self.connect()

    def connect(self):
        self._connection = serial.Serial(
            '/dev/ttyACM99',
            self.baud,
            timeout=10,
            parity= 'N',
            bytesize=8,
            stopbits=1
        )
        if self._connection.isOpen():
            self._connection.close()
        self._connection.open()

    def prepare_data(self, data):
        if len(data) > MAX_LEDS:
            raise ValueError("%s exceeds MAX_LEDS" % len(data))

        self.data = data

    def write(self,val):
        self._connection.write(val)
        
    def read(self,n):
        return self._connection.read(n)

    def write_whole_array(self):
        arr = []
        if self.debug:
            print("Writing data...")

        length = len(self.data)

        start = time.time()
        self.connect()
        self._connection.write(chr((length >> 8) & 0xff))
        self._connection.write(chr(length & 0xff))

        for idx in range(length):
            self._connection.write(chr(self.data[idx]))
        self._connection.write('\0')

        if self.debug:
            print("Time to write: %s" % (time.time() - start))

        arr += [self._connection.read()]
        
        arr += self.read(self._connection.inWaiting())
        print(arr)

        end = time.time()
        if self.debug:
            print("Time taken: %s" % (end - start))
            print(arr)

        self.close()
        return arr

    def close(self):
        self._connection.close()

    def readline(self):
        self._lastresponse = self._connection.readline()
        return self._lastresponse


SIMULATOR_SETUP = False

class UartSimulator:
    def __init__(self, debug=False):
        self.debug = debug
        self.data = []
        global SIMULATOR_SETUP
        if not SIMULATOR_SETUP:
            self._setup()

    def _setup(self):
        global SIMULATOR_SETUP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', settings.UART_SIM_PORT))

            s.sendall(struct.pack(
                settings.SIM_PARAM_STRUCT,
                settings.SIM_COLS,
                settings.SIM_ROWS,
                settings.SIM_EDGE_LENGTH,
                settings.SIM_DOWNWARDS,
                settings.SIM_RIGHTWARDS,
                settings.SIM_HORIZONTAL,
                settings.SIM_DRAW_CONNECTORS,
                settings.SIM_FPS
            ))
        SIMULATOR_SETUP = True

    def prepare_data(self, data):
        self.data = data

    def write_whole_array(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', settings.UART_SIM_PORT))
        
            idx = 0
            while idx < len(self.data):
                cache = []
                for col in self.data[idx:idx + 3]:
                    for bit in range(7, -1, -1):
                        cache.append((col >> bit) & 1)
                idx += 3
                s.sendall(struct.pack(settings.SIM_LED_STRUCT, *cache,
                                      idx >= len(self.data)))
            