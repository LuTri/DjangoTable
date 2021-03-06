import serial
import sys
import csv

MAX_LEDS = 336

class UartCom(object):
	def __init__(self):
		self._connection = serial.Serial('/dev/ttyACM99',9600,timeout=5)
		if self._connection.isOpen():
			self._connection.close()
		self._connection.open()
		self.data = [None] * MAX_LEDS

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
		self._connection.write(chr((length >> 8) & 0xff))
		self._connection.write(chr(length & 0xff))

		for idx in range(length):
			self._connection.write(chr(self.data[idx]))

		outcome = self._connection.readline()

	def close(self):
		self._connection.close()

	def readline(self):
		self._lastresponse = self._connection.readline()
		return self._lastresponse
