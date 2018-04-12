import serial
import time

MAX_LEDS = 336

class UartCom(object):
    def __init__(self, debug=False, baud=38400):
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
            print "Writing {} bytes of data...".format(len(self.data))

        length = len(self.data)

        start = time.time()
        self._connection.write(chr((length >> 8) & 0xff))
        self._connection.write(chr(length & 0xff))

        self._connection.write(self.data)

        while self._connection.out_waiting != 0:
            print "Waiting for output"

        if self.debug:
            print "Time to write: %s" % (time.time() - start)
        self.read(self._connection.in_waiting)
        while (self._connection.in_waiting != 0):
            print "SOMETHING'S WAITING FOR ya!"
            print self.read(self._connection.in_waiting)

        end = time.time()
        if self.debug:
            print "Time taken: %s" % (end - start)

        return arr

    def close(self):
        self._connection.close()

    def readline(self):
        self._lastresponse = self._connection.readline()
        return self._lastresponse
