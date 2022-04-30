import serial
import time

import socket
import struct

from django.conf import settings

MAX_LEDS = 336

class UartCom(object):
    STRUCT_FORMAT = '>3sHH{length}B'

    def __init__(self, debug=False, baud=38400):
        self.baud = baud
        self.data = []
        self.debug = debug
        self.connect()

    @property
    def fletcher_checksum(self):
        sum1 = sum2 = 0

        for x in self.data:
            sum1 = (sum1 + x) % 255
            sum2 = (sum2 + sum1) % 255

        return (sum1 << 8) | sum2

    @property
    def uart_out(self):
        length = len(self.data)
        return struct.pack(
            self.STRUCT_FORMAT.format(length=length),
            'TAD'.encode(),
            length,
            self.fletcher_checksum,
            *self.data
        )

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
            print("Writing {} bytes of data...".format(len(self.data)))

        length = len(self.data)

        start = time.time()

        self._connection.write(self.uart_out)

        #self._connection.write(chr((length >> 8) & 0xff))
        #self._connection.write(chr(length & 0xff))

        #self._connection.write(self.data)

        while self._connection.out_waiting != 0:
            print("Waiting for output")

        if self.debug:
            print("Time to write: %s" % (time.time() - start))
        self.read(self._connection.in_waiting)
        while (self._connection.in_waiting != 0):
            print("SOMETHING'S WAITING FOR ya!")
            print(self.read(self._connection.in_waiting))

        end = time.time()
        if self.debug:
            print("Time taken: %s" % (end - start))

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
