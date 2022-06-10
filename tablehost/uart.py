import logging
import random
import serial
import struct
import threading
import time
import importlib

from copy import deepcopy
from datetime import datetime
from functools import wraps
from pydoc import locate

from libs.remoteObj import TcpClient
from libs import mcconversion
from django.conf import settings
from .mcconfig import FileConfig


random.seed(datetime.now())

MAX_LEDS = 336

PARITY_MAPPING = {
    0: 'N',
    1: 'O',
    2: 'E'
}


class UartInterpreter:
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger('uart_threads')
        self.defines = FileConfig(filename=settings.UART_DEFINES_FILE)

    def code_for(self, msg):
        return self.defines.get(msg).encode()

    def handles(self, data):
        raise NotImplementedError

    def interpret(self, data):
        raise NotImplementedError


class VerboseInterpreter(UartInterpreter):
    def handles(self, data):
        return True

    def _split(self, data):
        found = []
        for key in [key for key in self.defines.keys()
                    if key.startswith('MSG_')]:
            code = self.code_for(key)
            if code in data:
                found.append((code, data.index(code)))

        if found:
            first = min(found, key=lambda x: x[1])
            return first[0], *data.split(first[0], 1)
        return b'', data, None

    def interpret(self, data):
        code, _data, remainder = self._split(data)
        try:
            self.logger.debug(f'Verbose data: "{code}" {_data.decode("utf-8")}')
        except Exception:
            self.logger.debug(f'Verbose data: "{code}" {_data}')

        return remainder


class BenchmarkInterpreter(UartInterpreter):
    _STARTER = 'MSG_BENCHMARK_START'
    _STOPPER = 'MSG_BENCHMARK_STOP'
    _DATA_STARTER = 'MSG_BENCHMARK_DATA'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = False
        self._raw = bytearray()

    @property
    def STARTER(self):
        return self.code_for(self._STARTER)

    @property
    def STOPPER(self):
        return self.code_for(self._STOPPER)

    @property
    def DATA_STARTER(self):
        return self.code_for(self._DATA_STARTER)

    def handles(self, data):
        if self._active:
            return True
        if data.startswith(self.STARTER):
            self._active = True
            return True
        return False

    def add_data(self, data):
        self._raw.extend(data)

    @staticmethod
    def binary_single(value):
        val = []
        for item in value:
            val.append(f'{bin(item)[2:]:>08}')
        return '.'.join(val)

    @staticmethod
    def interpret_data_single(value):
        val = 0
        for idx, item in enumerate(value):
            val += (item << (8 * (len(value) - (idx + 1))))
        return val

    def _get_benchmarks(self):
        results = []

        data = self._raw.strip(self.STARTER)
        data = data.strip(self.STOPPER).strip(self.DATA_STARTER)

        unique_bms = data.split(self.DATA_STARTER + self.DATA_STARTER)

        for pair in unique_bms:
            _len = int(len(pair) / 2)
            isrs_start = self.interpret_data_single(pair[_len - 2:_len])
            isrs_end = self.interpret_data_single(pair[_len + 2:_len + 4])
            _start, _end = (pair[:_len - 4], pair[_len + 4:])

            v_start = self.interpret_data_single(_start)
            v_end = self.interpret_data_single(_end)

            self.logger.debug(f'End: {v_end}; Start: {v_start}; diff: '
                              f'{v_end - v_start}'
                              f' - ISRS: S:{isrs_start} E:{isrs_end}')
            self.logger.debug(f'{self.binary_single(_end)} - ')
            self.logger.debug(f'{self.binary_single(_start)}')

            results.append(v_end - v_start)
        return results

    def process(self):
        b_marks = sorted(self._get_benchmarks())

        _m_count = None
        _mode = {}

        mean = None
        for item in b_marks:
            mean = (mean + item) / 2 if mean is not None else item

            _mode.setdefault(item, 0)
            _mode[item] += 1

        modes = []
        for item in sorted(_mode, key=lambda k: _mode[k]):
            if _m_count is None:
                if _m_count == 1:
                    break
                _m_count = _mode[item]
            elif _mode[item] != _m_count:
                break
            modes.append(item)

        self.logger.info(
            f'BENCHMARK: MEAN: {mean}, '
            f'MEDIAN: {b_marks[int(len(b_marks) / 2)]}, '
            f'MODES: {sorted(modes)}, '
            f'RANGE: {max(b_marks) - min(b_marks)}'
        )
        self.logger.info(f'Fully processed benchmark')
        self.logger.debug(f'raw data was:')
        self.logger.debug(f'{[bytes([i]) for i in self._raw]}')
        del self._raw
        self._raw = bytearray()
        self._active = False

    def interpret(self, data):
        _data, remains = data, b''

        if self.STOPPER in data:
            _data, remains = _data.split(self.STOPPER)
            _data += self.STOPPER

        self.add_data(_data)
        if self.STOPPER in data:
            self.process()

        return remains


def flat_traceback(stack, beginner='line 3397, in run_code', always=False):
    if not always:
        return ''
    res = None

    _stack = stack
    while len(_stack) > 0:
        current = _stack.pop()
        if beginner in current:
            break
        res = res + current if res is not None else '\n' + current

    return res


class MonitorMixin:
    class PollerThread(threading.Thread):

        @staticmethod
        def requires_thread(fnc):
            @wraps(fnc)
            def wrapper(instance, *args, **kwargs):
                if instance._monitor_thread is not None:
                    if not instance._monitor_thread.is_alive():
                        instance.logger.warning(
                            f'MONITORING THREAD WAS DEAD (from fnc: '
                            f'{fnc.__name__} of {instance.__class__.__name__} '
                            f'{hex(id(instance))} '
                            #f'{flat_traceback(traceback.format_stack())})'
                        )
                        instance._new_thread()
                        if not instance.isOpen():
                            instance.open()
                else:
                    instance.logger.warning(
                        f'MONITORING THREAD WAS DEAD (from fnc: '
                        f'{fnc.__name__} of {instance.__class__.__name__} '
                        f'{hex(id(instance))})'
                        #f'{flat_traceback(traceback.format_stack())}'
                    )
                    instance._new_thread()
                    if not instance.isOpen():
                        instance.open()

                return fnc(instance, *args, **kwargs)
            return wrapper

        def __del__(self):
            self.logger.warning(
                f'Destroying {self}'
                #f' {flat_traceback(traceback.format_stack())}'
            )

        def __init__(self, monitor_instance, interval=0.1, daemon=True,
                     plugin_classes=None, *args, **kwargs):
            super().__init__(*args, daemon=daemon, **kwargs)

            self.plugins = [_cls() for _cls in plugin_classes or
                            [BenchmarkInterpreter, VerboseInterpreter]]

            self.logger = logging.getLogger('uart_threads')
            self.monitor_instance = monitor_instance
            self.alive = True
            self.interval = interval

        def interpret(self, data):
            remainder = data
            while remainder:
                handled = False
                for plugin in self.plugins:
                    if plugin.handles(remainder):
                        handled = True
                        remainder = plugin.interpret(remainder)
                        break
                if not handled:
                    self.logger.warning(
                        f'Could not interpret {remainder}'
                        #f' {flat_traceback(traceback.format_stack())}'
                    )
                    break

        def run(self):
            self.logger.info(
                f'Starting monitoring thread.'
                #f' {flat_traceback(traceback.format_stack(), always=True)}'
            )
            while self.alive:
                time.sleep(self.interval)
                if not self.monitor_instance.isOpen():
                    continue

                _waiting = self.monitor_instance.s_in_waiting
                if _waiting:
                    data = self.monitor_instance.s_read(_waiting)
                    try:
                        self.interpret(data)
                    except Exception as exc:
                        self.logger.error(exc)
                    self.monitor_instance._push_bytes(data)
            self.logger.error(
                f'Thread reached END'
                #f' {flat_traceback(traceback.format_stack())}'
            )

        def stop(self):
            self.alive = False
            self.join()
            self.monitor_instance = None

    def _get_waiting(self):
        #self.logger.info('Acquiring lock.')
        with self._buffer_lock:
            _len = len(self._buffer)
            #self.logger.info(f'Acquired. {_len=}')
            return _len

    def _get_bytes(self, n=None):
        #self.logger.info(f'Acquiring lock. ({n=})')
        with self._buffer_lock:
            #self.logger.info('Acquired.')
            n = n or len(self._buffer)
            out, self._buffer = self._buffer[:n], self._buffer[n:]
            #self.logger.info(f'Acquired. {out=}, {self._buffer=}')
            return out

    def _push_bytes(self, data):
        #self.logger.info('Acquiring lock.')
        with self._buffer_lock:
            #self.logger.info('Acquired.')
            self._buffer.extend(data)

    @property
    def s_in_waiting(self):
        return super().in_waiting

    def s_read(self, n):
        return super().read(n)

    @PollerThread.requires_thread
    @property
    def out_waiting(self):
        return super().out_waiting

    @PollerThread.requires_thread
    def write(self, *args, **kwargs):
        return super().write(*args, **kwargs)

    @PollerThread.requires_thread
    @property
    def in_waiting(self):
        return self._get_waiting()

    @PollerThread.requires_thread
    def read(self, n=0):
        #self.logger.info(f'Attempting read. ({n=})')
        _out = bytearray()
        while True:
            _out += self._get_bytes(n)
            if len(_out) >= n:
                break
            time.sleep(.1)
        return _out

    def close(self):
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self.logger.warning(
                f'Stopping Monitor thread!'
                #f' {flat_traceback(traceback.format_stack())}'
            )
            self._monitor_thread.stop()
        super().close()

    def reload(self):
        self.close()
        self.open()

    def __del__(self):
        self._monitor_thread.stop()
        self.close()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger('uart_threads')
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._monitor_thread = None

    def _new_thread(self):
        self._monitor_thread = self.PollerThread(self)
        self._monitor_thread.start()


class PatchedSerial(serial.Serial):

    def outWaiting(self):
        return self.out_waiting

    def read_until(self, expected, size=None):
        """\
        Read until an expected sequ ence is found ('\n' by default), the size
        is exceeded or until timeout occurs.
        """
        logger = logging.getLogger('uart_com')
        logger.warning(
            'READ_UNTIL IN OVERRIDEN METHOD!'
            #f' {flat_traceback(traceback.format_stack())}'
        )
        if not type(expected) in [tuple, list, set]:
            expected = [expected]

        lenterm = [len(e) for e in expected]
        line = bytearray()
        timeout = serial.Timeout(self._timeout)
        while True:
            expected_found = False
            c = self.read(1)
            if c:
                line += c
                for _len, _expected in zip(lenterm, expected):
                    if line[-_len:] == _expected:
                        expected_found = True
                        break
                if expected_found:
                    break
                if size is not None and len(line) >= size:
                    break
            else:
                break
            if timeout.expired():
                break
        return bytes(line)


class MonitoringSerial(MonitorMixin, PatchedSerial):
    pass


if settings.UART_TCP_WRAP:
    DEFAULT_SERIAL_CLS = TcpClient.wrap(PatchedSerial)
else:
    DEFAULT_SERIAL_CLS = PatchedSerial


class UartError(Exception):
    pass


class UartReadTimeout(UartError):
    pass


class UartConfigurationError(UartError):
    pass


class UartCom(object):
    DEFINES = FileConfig(filename=settings.UART_DEFINES_FILE)

    CMD = NotImplemented
    STRUCT_FORMAT = '>3sBHH{length}B4s'

    def __init__(self, timeout=settings.UART_TIMEOUT,
                 ini_filename=settings.UART_INI_FILE,
                 serial_class=None):

        self._logger = logging.getLogger('uart_com')

        self._config = FileConfig(
            filename=ini_filename,
            defaults={
                'BAUD': settings.UART_BAUD_RATE,
                'MC_CLOCK': None,
                'PORT': settings.UART_PORT,
                'CHAR_SIZE': settings.UART_BYTESIZE_DEFAULT,
                'STOP_BITS': settings.UART_STOPBITS_DEFAULT,
                'PARITY_MODE': settings.UART_PARITY_MODE_DEFAULT,
            },
            file_traverse_order=['USART0', 'USART1', 'USART2', 'USART3'],
            bubble_reload_to=self,
            map_get_to=None,
        )

        module, klass = settings.LOCAL_SERIAL_CLASS.rsplit('.', 1)
        module = importlib.import_module(module)

        self.__expected_answers = None
        self._serial_class = serial_class
        if self._serial_class is None:
            self._serial_class = DEFAULT_SERIAL_CLS or getattr(module, klass)
        self.__connection = None

        self.data = []

        self._timeout = timeout

    def _purge_connection(self):
        if self.__connection is not None:
            self._logger.warning(
                f'Purging self.__connection ('
                f'{self.__connection.__class__.__name__} '
                f'{hex(id(self.__connection))}) of {self}'
            )
            self.__connection.close()
            del self.__connection
            self.__connection = None
        else:
            self._logger.warning(
                f'Purging self.__connection '
                f'({self.__connection}) of {self}'
                #f' {flat_traceback(traceback.format_stack())}'
            )

    def reload(self):
        self._logger.info(
            f'Reloading __connection and __expected_answers of {self}.'
            #f' {flat_traceback(traceback.format_stack())}'
        )
        self.connection.reload()
        self.__expected_answers = None

    @property
    def connection(self):
        if self.__connection is None:
            self._logger.error(
                f'PROPERTY .connection: Creating new Serial. {self}.'
                #f' {flat_traceback(traceback.format_stack())}'
            )
            self.__connection = self._serial_class(
                port=self._config.get('PORT'),
                baudrate=self._config.get('BAUD'),
                timeout=self._timeout,
                parity=PARITY_MAPPING.get(self._config.get('PARITY_MODE'),
                                          self._config.get('PARITY_MODE')),
                bytesize=self._config.get('CHAR_SIZE'),
                stopbits=self._config.get('STOP_BITS'),
            )
        self._logger.error(
            f'ACCESSED .connection: {self}.'
            #f' {flat_traceback(traceback.format_stack())}'
        )
        return self.__connection

    def mc_cycles_between_bytes(self):
        _speed = self._config.get('MC_CLOCK', None)
        if _speed is None:
            raise RuntimeError('Could not determine MC speed.')
        cycles_per_bit = _speed / self._config.get('BAUD')
        _byte_size = (self._config.get('STOP_BITS') +
                      self._config.get('CHAR_SIZE'))
        return cycles_per_bit * _byte_size

    def __del__(self):
        self._purge_connection()

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
            self.DEFINES.get('UART_HEADER_BYTES').encode(),
            self.DEFINES.get(self.CMD),
            length,
            self.fletcher_checksum,
            *self.data,
            self.DEFINES.get('UART_DONE_BYTES').encode(),
        )

    def connect(self):
        if not self.connection.isOpen():
            self.connection.open()

    def close(self):
        self._logger.info(
            f'Closing UartCom instance {self}'
            #f' {flat_traceback(traceback.format_stack())}'
        )
        self._purge_connection()

    def prepare_data(self, data):
        self._logger.debug(
            f'{self.__class__.__name__}: setting internal data '
            f'= {data} '
            #f'{flat_traceback(traceback.format_stack())}'
        )
        if len(data) > MAX_LEDS:
            raise ValueError("%s exceeds MAX_LEDS" % len(data))

        self.data = data

    @property
    def expected_answers(self):
        if self.__expected_answers is None:
            expected = self.DEFINES.get('MSG_ANSWER_START').encode()

            values = [
                expected + self.DEFINES.get('MSG_BUFFER_OVERFLOW').encode(),
                expected + self.DEFINES.get('MSG_DATA_OVERRUN').encode(),
                expected + self.DEFINES.get('MSG_FRAME_ERROR').encode(),
                expected + self.DEFINES.get('MSG_CHECKSUM_ERROR').encode(),
                expected + self.DEFINES.get('MSG_OK').encode(),
            ]
            for value in values:
                for other in values:
                    if other != value and (value in other or other in value):
                        raise UartConfigurationError(
                            f'Ambiguous answers expected: {value} {other}'
                        )
            self.__expected_answers = values
        return self.__expected_answers

    def _write(self):
        n_bytes = self.connection.write(self.uart_out)
        self._logger.info(
            f'Wrote {n_bytes}bytes to UART.'
            #f' {flat_traceback(traceback.format_stack())}'
        )
        self._logger.debug(
            f'Data: {self.uart_out}'
            #f'{flat_traceback(traceback.format_stack())}'
        )
        reply = self.connection.read_until(expected=self.expected_answers)

        if not any(exp in reply for exp in self.expected_answers):
            raise UartReadTimeout(f'{self._config.get("PORT")} read Timeout. Received: {reply}')
        return self.parse_reply(reply), reply

    def parse_reply(self, reply):
        messages = []
        parts = reply.split(self.DEFINES.get('MSG_ANSWER_START').encode())
        for part in parts[1:]:
            messages.append(self.DEFINES.reverse(part))
        return messages

    @staticmethod
    def _process_param(_param, definition):
        _range_op_mapping = {
            'r_left_inc': lambda a, b: a >= b,
            'r_left_exc': lambda a, b: a > b,
            'r_right_inc': lambda a, b: a <= b,
            'r_right_exc': lambda a, b: a < b,
        }

        _type = locate(definition['type'])
        _conversion = definition.get('conversion', None)

        for key, op in _range_op_mapping.items():
            try:
                if not op(_param, definition[key]):
                    raise ValueError(f'{_param} is not in specified range for '
                                     f'{definition["name"]}!')
            except KeyError:
                pass

        _value = _type(_param)
        if _conversion is not None:
            converter = getattr(mcconversion, _conversion)
            return converter(_param)

        return [_value]

    def _validate_params(self, *args, **kwargs):
        _list_args = list(args)
        _dict_kwargs = {}
        _dict_kwargs.update(kwargs)

        _arg_pop = []

        _possible = deepcopy(self.DEFINES.data.get(self.CMD, {}).get('params', []))
        _valid = {}

        # Validate any existing keyword argument against possible params
        if _dict_kwargs:
            for idx, _def in enumerate(_possible):
                if _def['name'] in _dict_kwargs:
                    _valid[idx] = self._process_param(_dict_kwargs[_def['name']], _def)
                    del _dict_kwargs[_def['name']]

            for _idx in reversed(sorted(_valid)):
                _possible.pop(_idx)

        # find possible positional arguments for possible params
        _valid_idx = 0
        for idx, pair in enumerate(zip(_list_args, _possible)):
            _in, _def = pair

            while _valid_idx in _valid:
                _valid_idx += 1

            _valid[_valid_idx] = self._process_param(_in, _def)
            _arg_pop.append(idx)

        for _idx in reversed(_arg_pop):
            _list_args.pop(_idx)
            _possible.pop(_idx)

        _arg_pop = []
        _valid_idx = 0

        # set default values for all remaining possible params if applicable
        for _idx, _def in enumerate(_possible):
            if _def.get('default', None) is not None:
                while _valid_idx in _valid:
                    _valid_idx += 1
                _valid[_valid_idx] = self._process_param(_def['default'], _def)
                _arg_pop.append(_idx)

        for _idx in reversed(_arg_pop):
            _possible.pop(_idx)

        if _dict_kwargs:
            raise ValueError(f'Unexpected keyword arguments: {list(_dict_kwargs.keys())}')
        if _list_args:
            raise ValueError(f'Unexpected positional arguments: {_list_args}')
        if _possible:
            raise ValueError(f'Missing arguments: {[p["name"] for p in _possible]}')

        return [_byte for key in sorted(_valid) for _byte in _valid[key]]

    def command(self, *args, **kwargs):
        _parameters = self._validate_params(*args, **kwargs)
        self.prepare_data(_parameters)
        return self._write()


class UartChecksumErrorTest(UartCom):
    CMD = 'CMD_SLAVE'

    @property
    def fletcher_checksum(self):
        return 0xFFFF

    def test(self):
        test_data = [1, 3, 5, 2, 4, 6]
        self.prepare_data(test_data)

        return self._write()


class UartSlave(UartCom):
    CMD = 'CMD_SLAVE'

    def command(self, *data, **kwargs):
        if len(data) == 1 and len(data[0]) > 1:
            data = [item for item in data[0]]
        self.prepare_data(data)
        return self._write()


class UartSnake(UartCom):
    CMD = 'CMD_SNAKE'


class UartMood(UartCom):
    CMD = 'CMD_MOOD'


class SimpleUartCmd(UartCom):
    def command(self, *args, **kwargs):
        return self._write()


class UartWhite(SimpleUartCmd):
    CMD = 'CMD_WHITE'


class UartOff(SimpleUartCmd):
    CMD = 'CMD_OFF'


class UartReboot(SimpleUartCmd):
    CMD = 'CMD_REBOOT'


class UartSetDoBenchmark(UartCom):
    CMD = 'CMD_SET_DO_BENCHMARK'


class LedWriter(UartCom):
    CMD = 'CMD_WRITE'


class SoundToLight(UartCom):
    CMD = 'CMD_SOUNDTOLIGHT'

    def __init__(self, *args, os_new_min=None, os_new_max=None,
                 os_old_min=None, os_old_max=None, method=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.os_old_min = os_old_min
        self.os_old_max = os_old_max
        self.os_new_min = os_new_min
        self.os_new_max = os_new_max
        self.combiner = method

    def handle(self, frequency_domain_data, *args):
        if not self.connection.is_open:
            self.connection.open()
        frequency_domain_data.write(self,
                                    method=self.combiner,
                                    o_scale_new_min=self.os_new_min,
                                    o_scale_new_max=self.os_new_max,
                                    o_scale_old_min=self.os_old_min,
                                    o_scale_old_max=self.os_old_max)
        self.connection.close()


class UartSetState(UartCom):
    CMD = 'CMD_SET_STATE'


class UartGetState(UartCom):
    CMD = 'CMD_GET_STATE'

    @property
    def expected_answers(self):
        return [self.DEFINES.get('MSG_STATE_DATA_STOP').encode(),]

    def command(self, *args, **kwargs):
        from libs.mcconversion import per_one_2byte
        from libs.mcconversion import dualbyte
        from libs.mcconversion import full_float

        reply, current_state = super().command(*args, **kwargs)

        garbage, data = current_state.split(b'SD')
        data = data.rstrip(b'DS')

        results = {
            'load_status': 'LOADED' if int(data[0]) == 1 else 'INITIALIZED',
            'intensity': per_one_2byte.reverse(data[1:3]),
            'fnc_count': dualbyte.reverse(data[3:5]),
            'dim_delay': dualbyte.reverse(data[5:7]),
            'hues': [full_float.reverse(data[7 + i * 4: 7 + (i + 1) * 4]) for i in range(8)],
        }

        return reply, results
