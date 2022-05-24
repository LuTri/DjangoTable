import socket
import struct
import threading
import logging

import numpy as np


class PCMData:
    def __init__(self, data, channels, sample_rate, n_samples, frame_counter):
        self.channels = channels
        self.sample_rate = sample_rate
        self.frame_counter = frame_counter
        self.n_samples = n_samples

        t_data = np.array(data)
        self.data = np.reshape(
            t_data,
            (int(t_data.size / self.channels), self.channels)
        )

    def mean_channels(self):
        return np.mean(np.swapaxes(self.data, 1, 0), 0)

    def mean_channels(self):
        return np.mean(np.swapaxes(self.data, 1, 0), 0)


def signed_24(bytes):
    prefix = 0xFF if bytes[0] >= 128 else 0x00
    return struct.unpack('i', struct.pack('BBBB', prefix, *bytes))


class VBANReceiver(threading.Thread):
    HEADER_FORMAT = struct.Struct('4sBBBB16sI')
    SR_LIST = [6000, 12000, 24000, 48000, 96000, 192000, 384000, 8000, 16000,
               32000, 64000, 128000, 256000, 512000, 11025, 22050, 44100,
               88200, 176400, 352800, 705600]

    SUBPROTOCOLS = [
        'AUDIO',
        'SERIAL',
        'TXT',
        'SERVICE',
        'UNDEFINED',
        'UNDEFINED',
        'UNDEFINED',
        'USER',
    ]

    DATA_CODECS = {
        0x0: 'PCM',
        0x1: 'VBCA',
        0x2: 'VBCV',
        0xF: 'USER',
    }

    DATA_FORMATS = [
        ('8bits unsigned', 'B', int),
        ('16bits signed', 'h', int),
        ('24bits signed', None, None),
        ('32bits signed', 'i', int),
        ('32bits float', 'f', float),
        ('64bits float', 'd', float),
        ('12bits signed', None, None),
        ('10bits signed', None, None),
    ]

    def __init__(self, caller, senderIp=None, port=52000,
                 stream='SheepdroidVban', verbose=False, daemon=True, *args,
                 **kwargs):
        super().__init__(*args, daemon=daemon, **kwargs)

        self.rec_couner = 0

        self.caller = caller
        self.logger = logging.getLogger('uart_threads')

        self.streamName = stream
        self.senderIp = senderIp

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(2.0)
        self.sampRate = 48000
        self.channels = 2
        self.dataFormat = 2
        self.stream_magicString = ""
        self.stream_sampRate = 0
        self.stream_sampNum = 0
        self.stream_chanNum = 0
        self.stream_dataFormat = self.DATA_FORMATS[0]
        self.stream_streamName = ""
        self.stream_frameCounter = 0
        self.stream_Codec = None

        self.rawPcm = None
        self.running = True
        self.verbose = verbose

        self.rawData = None
        self.subprotocol = 0
        self.logger.info(f'Fully initialized {self.__class__.__name__}')

    def _cutAtNullByte(self, stri):
        return stri.decode('utf-8').split("\x00")[0]

    def _parse_header(self, data):
        vban, sr, n_samples, n_channels, df, stream_name, frame_nr = self.HEADER_FORMAT.unpack(data[:self.HEADER_FORMAT.size])
        self.logger.info(
            f'Welcome: {vban}, sub protocol: {sr} n samples: {n_samples} n channels: {n_channels}, data format: {df} stream: {stream_name} frame nr: {frame_nr}'
        )

    def parse_header(self, data):
        _header = self.HEADER_FORMAT.unpack(data[:self.HEADER_FORMAT.size])

        self.stream_magicString = _header[0].decode('utf-8')
        self.subprotocol = self.SUBPROTOCOLS[(_header[1] & 0b11100000) >> 5]
        self.stream_sampRate = self.SR_LIST[_header[1] & 0b00011111]
        self.stream_sampNum = _header[2] + 1
        self.stream_chanNum = _header[3] + 1
        self.stream_dataFormat = self.DATA_FORMATS[_header[4] & 0b111]
        self.stream_Codec = self.DATA_CODECS.get(
            (_header[4] & 0b11110000) >> 4,
            'UNDEFINED'
        )
        self.stream_streamName = self._cutAtNullByte(_header[5])
        self.stream_frameCounter = _header[6]
        if self.verbose:
            self.logger.info(
                f'Welcome: {self.stream_magicString}, '
                f'Sample rate: {self.stream_sampRate} '
                f'Subprotocol: {self.subprotocol}'
                f'n samples: {self.stream_sampNum} '
                f'n channels: {self.stream_chanNum}, '
                f'data format: {self.stream_dataFormat} '
                f'Codec: {self.stream_Codec}'
                f'Streamname: {self.stream_streamName} '
                f'frame nr: {self.stream_frameCounter}'
            )

    def runonce(self):
        if not self.running:
            self.logger.error(
                f'{self.__class__.__name__} not running, yet runonce was '
                'called. '
            )
            return

        # buffer size is normally 1436 bytes Max size for vban
        data, addr = self.sock.recvfrom(2048)
        self.rawData = data
        self.parse_header(data)

        data_struct = struct.Struct(
            '<' + self.stream_dataFormat[1] * self.stream_sampNum *
            self.stream_chanNum
        )

        self.caller.push_pcm(PCMData(
            np.array(data_struct.unpack(data[self.HEADER_FORMAT.size:])),
            self.stream_chanNum,
            self.stream_sampRate,
            self.stream_sampNum,
            self.stream_frameCounter,
        ))

        self.rec_couner += 1
        if self.rec_couner % 6000 == 0:
            self.logger.info(f'Update: received {self.rec_couner} frames.')

    def run(self):
        while self.running:
            try:
                self.runonce()
            except socket.timeout:
                continue

    def quit(self):
        self.running = False
        self.join()


class VBANCollector:
    def __init__(self, verbose=False, visual=False, buffer_frames=100,
                 average_over=1, skip_frames=25, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._args = args
        self._kwargs = kwargs
        self._verbose = verbose
        self._visual = visual

        self._average_over = average_over
        self._skip_frames = skip_frames

        self.logger = logging.getLogger('uart_threads')
        self.buffer_frames = buffer_frames

        self.receiver_thread = None
        self.plotter_thread = None

        self.frame_lock = threading.Lock()
        self.__last_pcm = []
        self.__plot_fncs = {}
        self.plot_fnc_lock = threading.Lock()
        self.__parts = 14
        self.frame_counter = None

        self._total_frames_pushed = 0
        self.__apply_modulators = False

    def get_frame_counter(self):
        t = None
        if self.frame_lock.acquire(blocking=True, timeout=.5):
            t = self.frame_counter
            self.frame_lock.release()
        return t

    def last_n_pcm(self, n):
        t_val = []
        if self.frame_lock.acquire(blocking=True, timeout=.5):
            t_val = self.__last_pcm[-n:]
            self.frame_lock.release()
        return t_val

    def push_pcm(self, value):
        with self.frame_lock:
            self.frame_counter = value.frame_counter
            self.__last_pcm.append(value)
            if len(self.__last_pcm) >= self.buffer_frames:
                self.__last_pcm.pop(0)

            self._total_frames_pushed += 1
            #if self._total_frames_pushed % 100 == 0:
            #    self.logger.info(f'Pushed {self._total_frames_pushed} so far.')

    @property
    def parts(self):
        t_val = None
        if self.frame_lock.acquire(blocking=True, timeout=1):
            t_val = self.__parts
            self.frame_lock.release()
        return t_val

    @parts.setter
    def parts(self, value):
        with self.frame_lock:
            self.__parts = value

    @property
    def apply_modulators(self):
        t = None
        with self.plot_fnc_lock:
            t = self.__apply_modulators
        return t

    @apply_modulators.setter
    def apply_modulators(self, val):
        with self.plot_fnc_lock:
            self.__apply_modulators = val

    def run(self):
        self.receiver_thread = VBANReceiver(self, *self._args, verbose=self._verbose, **self._kwargs)
        self.plotter_thread = None
        if self._visual:
            from .testplotterthread import PyplotThread
            self.plotter_thread = PyplotThread(
                self, *self._args, average_over=self._average_over,
                skip_frames=self._skip_frames, **self._kwargs)

        self.receiver_thread.start()
        if self.plotter_thread is not None:
            self.plotter_thread.start()

    def __del__(self):
        if self.plotter_thread is not None:
            self.plotter_thread.running = False

    def quit(self):
        if self.receiver_thread is not None:
            self.receiver_thread.running = False
            self.receiver_thread.join()
        if self.plotter_thread is not None:
            self.plotter_thread.running = False
            self.plotter_thread.join()
