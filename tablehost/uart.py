import serial
import sys
import csv

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
		writer = csv.writer(open("timings.csv","w"))
		for idx in range(1,337):
			timings = []
			for rep in range(20):
			 	self._connection.write(chr((idx >> 8) & 0xff))
			 	self._connection.write(chr(idx & 0xff))
							 
				for x in range(idx):
					self._connection.write(chr(x % 10))
			
				overflows = self._connection.readline()
				timer_cnt = self._connection.readline()
				timings.append(int(overflows) * 0xffff + int(timer_cnt))
			foo = 0.0
			for i in range(20):
				foo = foo + float(timings[i])
			timing = float(foo) / float(20)
			secs = (1.0/16000000.0) * timing
			print "Average timing for %3d: %6.4f Sekunden -> %6.4f FPS" % (idx,secs,(1.0/secs),)
			writer.writerow((idx,timing,secs,(1.0/secs),))
		writer.close()
