import serial
import time

MAX_LEDS = 336

class UartCom(object):
	def __init__(self, debug=False, baud=500000):
		self.baud = baud
		self.data = [None] * MAX_LEDS
		self.debug = debug
		self.connect()

	def connect(self):
		self._connection = serial.Serial('/dev/ttyACM99', self.baud, timeout=5)
		if self._connection.isOpen():
			self._connection.close()
		self._connection.open()

	def prepare_data(self, data):
		if len(data) > MAX_LEDS:
			raise ValueError("%s exceeds MAX_LEDS" % len(data))

		for idx in range(len(data)):
			self.data[idx] = data[idx]		

	def write(self,val):
		self._connection.write(val)
		
	def read(self,n):
		return self._connection.read(n)

	def write_whole_array(self,length = MAX_LEDS):
		arr = []
		if self.debug:
			print "Writing data..."
		start = time.time()
		self._connection.write(chr((length >> 8) & 0xff))
		self._connection.write(chr(length & 0xff))

		for idx in range(length):
			self._connection.write(chr(self.data[idx]))

		arr.append(self._connection.read())

		while self._connection.inWaiting() > 0:
			arr.append(self._connection.read())

		end = time.time()
		if self.debug:
			print "Time taken: %s" % (end - start)

		return arr

	def close(self):
		self._connection.close()

	def readline(self):
		self._lastresponse = self._connection.readline()
		return self._lastresponse
