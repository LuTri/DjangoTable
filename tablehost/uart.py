import serial
import time

import socket
import struct

from datetime import datetime

from libs.remoteObj import TcpClient

from django.conf import settings

import random

random.seed(datetime.now())


MAX_LEDS = 336


class PatchedSerial(serial.Serial):
    def outWaiting(self):
        return self.out_waiting


if settings.UART_TCP_WRAP:
    serialClass = TcpClient.wrap(PatchedSerial)
else:
    serialClass = PatchedSerial


class UartCom(object):
    STRUCT_FORMAT = '>3sHH{length}B'

    def __init__(self, debug=False, baud=settings.UART_BAUD_RATE):
        self.baud = baud
        self.data = []
        self.debug = debug

        self._connection = serialClass(
            port=settings.UART_PORT,
            baudrate=self.baud,
            timeout=10,
            parity= 'E',
            bytesize=8,
            stopbits=1,
        )

    def test_echo(self):
        failures = []
        for idx in range(settings.UART_N_ECHO_TEST):
            data = bytes([random.randint(0x00, 0xff) for x in range(settings.UART_N_ECHO_TEST_BYTES)])
            self._connection.write(data)
            result = self._connection.read(self._connection.inWaiting())
            if result != data:
                failures.append((idx, data, result))
        return failures

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
            self.STRUCT_FORMAT.format(length=length + 1),
            'DAT'.encode(),
            length,
            self.fletcher_checksum,
            *self.data,
            0,
        )

    def connect(self):
        if self._connection.isOpen():
            self._connection.close()
        self._connection.open()

    def prepare_data(self, data):
        if len(data) > MAX_LEDS:
            raise ValueError("%s exceeds MAX_LEDS" % len(data))

        self.data = data

    def write_whole_array(self):
        if self.debug:
            print("Writing {} bytes of data...".format(len(self.data)))

        start = time.time()
        self._connection.write(self.uart_out)

        while self._connection.outWaiting() != 0:
            pass

        if self.debug:
            print("Time to write: %s" % (time.time() - start))

        arr = self._connection.read(self._connection.inWaiting())
        while self._connection.inWaiting():
            arr += self._connection.read(self._connection.inWaiting())

        end = time.time()
        if self.debug:
            print("Time taken: %s" % (end - start))

        return arr


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
