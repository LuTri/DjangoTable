from django.core.management import BaseCommand
from django.core.management import CommandError

from tablehost.uart import UartCom
import serial
import time
from random import randint

class Command(BaseCommand):
    _uart_con = UartCom(debug=True)
    _n_leds = 112

    def handle(self, *args, **options):
        for x in range(0,255,10):
            for y in range(3):
                col = [0,0,0]

                col[y] = x
                data = col * self._n_leds

                self._uart_con.prepare_data(data)
                self._uart_con.write_whole_array()
            time.sleep(5)
        self._uart_con.close()
