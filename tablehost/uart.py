import serial
import sys

class UartCom(object):
	def __init__(self):
		self._connection = serial.Serial('/dev/ttyACM99',9600,timeout=1)
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

	def close(self):
		self._connection.close()
