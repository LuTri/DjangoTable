import threading
import pickle
import struct
import sys
import socket

import inspect

from functools import wraps
from types import MethodType

from django.conf import settings


class PayloadCommunicator:
    TCP_START = 'B'.encode()
    TCP_DONE = 'E'.encode()

    FMT_N_TYPE = struct.Struct('1s1sI1s')
    FMT_SIZE = struct.Struct('1sI1s')
    FMT_OBJ = '1s{}s1s'

    TYPE_RESULT = 'R'.encode()
    TYPE_KWARG = 'K'.encode()
    TYPE_ARG = 'A'.encode()
    TYPE_FNC = 'F'.encode()

    class BrokenFrame(Exception):
        pass

    def __init__(self, _socket):
        self._socket = _socket

    def _pack(self, fmt, *payload, fmt_format=None):
        if not isinstance(fmt, struct.Struct):
            fmt = struct.Struct(fmt.format(*fmt_format))

        _data = fmt.pack(self.TCP_START, *payload, self.TCP_DONE)
        self._socket.sendall(_data)

    def _unpack(self, fmt, n_results=1, fmt_format=None):
        if not isinstance(fmt, struct.Struct):
            fmt = struct.Struct(fmt.format(*fmt_format))
        _dat = self._socket.recv(fmt.size)
        result = fmt.unpack(_dat)
        _start = result[0]
        if _start != self.TCP_START:
            raise self.BrokenFrame(f'Frame did not start with {self.TCP_START}: {_dat}')
        _end = result[-1]
        if _end != self.TCP_DONE:
            raise self.BrokenFrame(f'Frame did not end with {self.TCP_DONE}: {_dat}')

        _res = result[1:-1]
        if len(_res) != n_results:
            raise self.BrokenFrame(f'Expected {n_results} values, got {len(_res)} instead!')

        if n_results == 1:
            return _res[0]
        return _res

    def _unpack_n_type(self):
        return self._unpack(
            self.FMT_N_TYPE,
            2
        )

    def _unpack_size(self):
        return self._unpack(
            self.FMT_SIZE,
        )

    def _unpack_obj(self, size):
        obj_str = self._unpack(
            self.FMT_OBJ,
            fmt_format=(size,),
        )
        return pickle.loads(obj_str)

    def _pack_size(self, size):
        self._pack(self.FMT_SIZE, size)

    def _pack_obj(self, obj):
        obj_bytes = pickle.dumps(obj)
        size = len(obj_bytes)
        self._pack_size(size)
        self._pack(
            self.FMT_OBJ,
            obj_bytes,
            fmt_format=(size,),
        )

    def _pack_n_type(self, type_char, n=1):
        self._pack(self.FMT_N_TYPE, type_char, n)

    def send_result(self, **kwargs):
        self._pack_n_type(self.TYPE_RESULT, len(kwargs))
        for key, value in kwargs.items():
            self._pack_obj({key: value})

    def send_kwargs(self, **kwargs):
        self._pack_n_type(self.TYPE_KWARG, len(kwargs))
        for key, value in kwargs.items():
            self._pack_obj({key: value})

    def send_args(self, *args):
        self._pack_n_type(self.TYPE_ARG, len(args))

        for item in args:
            self._pack_obj(item)

    def send_fnc(self, fnc_name):
        self._pack_n_type(self.TYPE_FNC)
        self._pack_obj(fnc_name)

    def get_result(self):
        result = {}
        type_char, n = self._unpack_n_type()

        if type_char != self.TYPE_RESULT:
            raise self.BrokenFrame('Expected RESULT data!')

        for idx in range(n):
            size = self._unpack_size()
            _obj = self._unpack_obj(size)
            result.update(_obj)

        return result

    def get_kwargs(self, n=None):
        res = {}

        if n is None:
            type_char, n = self._unpack_n_type()
            if type_char != self.TYPE_KWARG:
                raise self.BrokenFrame('Expected KWARG data!')

        for idx in range(n):
            size = self._unpack_size()
            _obj = self._unpack_obj(size)
            res.update(_obj)

        return res

    def get_fnc(self):
        size = self._unpack_size()
        return self._unpack_obj(size)

    def get_args(self, n=None):
        args = []

        if n is None:
            type_char, n = self._unpack_n_type()
            if type_char != self.TYPE_ARG:
                raise self.BrokenFrame('Expected ARG data!')

        for idx in range(n):
            size = self._unpack_size()
            _obj = self._unpack_obj(size)
            args.append(_obj)

        return args

    def get_invocation(self):
        args = ()
        kwargs = {}
        fnc_name = None

        missing = [self.TYPE_FNC, self.TYPE_ARG, self.TYPE_KWARG]

        for x in range(len(missing)):
            type_char, n = self._unpack_n_type()
            missing.remove(type_char)

            if type_char == self.TYPE_ARG:
                args = self.get_args(n)
            elif type_char == self.TYPE_KWARG:
                kwargs = self.get_kwargs(n)
            elif type_char == self.TYPE_FNC:
                fnc_name = self.get_fnc()
            else:
                raise self.BrokenFrame(f'Unknown frame type {type_char}!')

        if missing:
            raise self.BrokenFrame(f'Missing data for full invocation: {missing}')

        return fnc_name, args, kwargs


class Echoer:
    def __init__(self):
        self._data = []
        self._recv_idx = None

    def sendall(self, data):
        self._data.append(data)
        #sys.stderr.write(f'{data}\n')

    def recv(self):
        if self._recv_idx is None:
            self._recv_idx = 0
        else:
            self._recv_idx += 1
        return self._data[self._recv_idx]


class TestClass:
    def __init__(self, foo, bar, has_stuff=False):
        self.foo = foo
        self.bar = bar
        if has_stuff:
            self.stuff = ['super', 'stuff']
        else:
            self.stuff = None

    def get_stuff(self):
        return self.stuff

    def get_bar(self):
        return self.bar

    def set_bar(self, value):
        self.bar = value


class SocketWrapper:
    def __init__(self, _socket):
        self._socket = _socket
        self._buffer = bytearray()

    def sendall(self, data):
        self._socket.sendall(data)

    def recv(self, n):
        while len(self._buffer) < n:
            _buf = self._socket.recv(1024)
            if not _buf:
                raise socket.error('Disconnected')
            self._buffer.extend(_buf)

        result = self._buffer[:n]
        self._buffer = self._buffer[n:]

        return result


class SerialReader(threading.Thread):
    def __init__(self, _socket, _server_instance=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alive = True
        self.communicator = PayloadCommunicator(_socket)
        self._server_instance = _server_instance

    def close(self):
        self.alive = False
        self.join(2)

    def run(self):
        #sys.stderr.write(f'{threading.get_ident()} (SerialReader): started!\n')

        while self.alive:
            try:
                fnc_name, args, kwargs = self.communicator.get_invocation()
                #sys.stderr.write(f'{threading.get_ident()} (SerialReader): {fnc_name=}; {args=}; {kwargs=}\n')

                try:
                    if fnc_name == '_constructor':
                        self._server_instance.construct_wrapped(*args, **kwargs)
                        self.communicator.send_result(success=True)
                    else:
                        data = getattr(self._server_instance, fnc_name)

                        if callable(data):
                            data = data(*args, **kwargs)

                        self.communicator.send_result(success=True, result=data)

                except Exception as e:
                    self.communicator.send_result(success=False, error=e)

            except socket.error as exc:
                #sys.stderr.write(f'{threading.get_ident()} (SerialReader): SOCKET ERR {exc}\n')
                self.alive = False
            except KeyboardInterrupt:
                #sys.stderr.write(f'{threading.get_ident()}: KeyboardInterrupt, shutting down.')
                self.alive = False

        #sys.stderr.write(f'{threading.get_ident()} (SerialReader): KILLED!\n')


class TcpWrapped:
    help = 'Simple Serial to Network (TCP/IP) redirector.'
    epilog = """\
    NOTE: no security measures are implemented. Anyone can remotely connect
    to this service over the network.

    Only one connection at once is supported. When the connection is terminated
    it waits for the next connect.
    """

    def __init__(self, remote_class=None, port=None, **kwargs):
        self._remote_class = remote_class
        self._remote_port = port
        self._threads = []

        #sys.stderr.write(f'extra kwargs: {kwargs}\n')

    @classmethod
    def add_arguments(cls, parser, default_class=None):
        parser.add_argument(
            '-p', '--port',
            default=7777,
        )

        parser.add_argument(
            '-c', '--class',
            dest='remote_class',
            default=default_class or TestClass,
        )
        parser.add_argument(
            '-r', '--remote',
            dest='remote_addr',
            default='127.0.0.1',
        )

    def __call__(self):
        try:
            self._run()
        except KeyboardInterrupt:
            pass

        #sys.stderr.write('\n--- exit ---\n')
        for thread in self._threads:
            thread.close()

    def _run(self):
        pass


class TcpClient(TcpWrapped):
    def __init__(self, remote_addr=None, **kwargs):
        super().__init__(**kwargs)
        self._remote_addr = remote_addr
        self._communicator = None

    def _connect(self):
        _socket = socket.socket()
        try:
            _socket.connect((self._remote_addr, int(self._remote_port)))
        except socket.error as msg:
            return True

        try:
            _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
            _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            _socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except AttributeError as exc:
            pass  # XXX not available on windows
            #sys.stderr.write(f'ERROR not available on windows: {exc=}')
        _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self._communicator = PayloadCommunicator(SocketWrapper(_socket))

    @classmethod
    def wrap(cls, other_cls):
        #print(f'{cls=}, {other_cls=}')

        def __other__init__(self, *args, **kwargs):
            #print(f'OTHER_INIT, {args=}, {kwargs=}')
            self.__tcp = cls(remote_addr=settings.UART_TCP_HOST,
                             port=settings.UART_TCP_PORT)

            self.__tcp._connect()
            self.__tcp._communicator.send_fnc('_constructor')
            self.__tcp._communicator.send_args(*args)
            self.__tcp._communicator.send_kwargs(**kwargs)
            result = self.__tcp._communicator.get_result()
            if not result.get('success', False):
                raise result.get(
                    'error',
                    RuntimeError('Could not initialize remotely!'),
                )

        def make_fnc(fnc_name):

            def fnc(self, *args, **kwargs):
                self.__tcp._communicator.send_fnc(fnc_name)
                self.__tcp._communicator.send_args(*args)
                self.__tcp._communicator.send_kwargs(**kwargs)
                result = self.__tcp._communicator.get_result()
                #print(f'result pre .get= {result}')
                return result.get('result', None)
            return fnc

        attributes = {'__init__': __other__init__}
        dont_add = [
            '__init__',
            '__enter__',
            '__exit__',
            '__repr__',
        ]

        def predicate(obj):
            return inspect.isfunction(obj) or inspect.isdatadescriptor(obj)

        for name, _fnc in inspect.getmembers(other_cls, predicate):
            if name not in dont_add:
                attributes.update({name: make_fnc(name)})

        return type(
            f'{other_cls.__name__}(RemoteWrapped)',
            other_cls.__bases__,
            attributes,
        )


class TcpServer(TcpWrapped):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._wrapped = None

    def construct_wrapped(self, *args, **kwargs):
        self._wrapped = self._remote_class(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._wrapped, item)

    def _run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('', self._remote_port))
        server.listen(1)

        while True:
            self._threads = []
            #sys.stderr.write('Waiting for connection on {}...\n'.format(self._remote_port))
            _socket, addr = server.accept()
            #sys.stderr.write('Connected by {}\n'.format(addr))

            # More quickly detect bad clients who quit without closing the
            # connection: After 1 second of idle, start sending TCP keep-alive
            # packets every 1 second. If 3 consecutive keep-alive packets
            # fail, assume the client is gone and close the connection.
            try:
                _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
                _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
                _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                _socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except AttributeError as exc:
                pass  # XXX not available on windows
                #sys.stderr.write(f'ERROR not available on windows: {exc=}')
            _socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            socket_wrapper = SocketWrapper(_socket)
            self._threads = [
                SerialReader(socket_wrapper, self),
            ]

            for thread in self._threads:
                thread.start()

            for thread in self._threads:
                thread.join()

            self._wrapped = None
