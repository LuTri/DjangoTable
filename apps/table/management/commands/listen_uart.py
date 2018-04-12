from django.core.management import BaseCommand
from django.core.management import CommandError

from tablehost.uart import UartCom
import serial
import time
from random import randint

foo = UartCom(debug=True)

class Command(BaseCommand):
	def handle(self, *args, **options):
		uart = UartCom(debug=True)
		while True:
			try:
				print uart.readline()
			except OSError:
				print "Resource unavailable, restarting after 3 sconds..."
				time.sleep(3)
				uart = UartCom(debug=True)
			except serial.serialutil.SerialException:
				print "Device disconnected, restarting after 3 seconds..."
				time.sleep(3)
				uart = UartCom(debug=True)
			except KeyboardInterrupt:
				break
