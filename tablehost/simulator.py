import socket
import struct

from django.conf import settings

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
