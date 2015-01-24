import serial
import sys

class UartCom(object):
	def __init__(self):
		self._connection = serial.Serial('/dev/ttyACM99',9600,timeout=5)
		if self._connection.isOpen():
			self._connection.close()
		self._connection.open()
		
	def write(self, val):
		self._connection.write(val)

	def read(self,n):
		return self._connection.read(n)

	def write_color(self,rgb):
		self._connection.write('f')
		self._connection.write(chr(rgb[0]))
		self._connection.write(chr(rgb[1]))
		self._connection.write(chr(rgb[2]))
		self._connection.read(1)

	def write_whole_array(self):
		length = 336
		self._connection.write(chr((length >> 8) & 0xff))
		self._connection.write(chr(length & 0xff))
		idx = 0
		while (idx < 336):
			self._connection.write(chr(128))
			idx = idx + 1

	def close(self):
		self._connection.close()

	def readline(self):
		self._lastresponse = self._connection.readline()
		return self._lastresponse

	def testing(self):
	 	self._connection.write(chr(0))
	 	self._connection.write(chr(20))
					 
		print self._connection.readline()
		for x in range(20):
			self._connection.write(chr(x % 10))

		bar = '1'
		while (bar != ''):
			bar = self._connection.readline()
			print bar

