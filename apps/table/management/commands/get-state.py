import os

from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartGetState

from libs.mcconversion import full_float
from libs.mcconversion import per_one_2byte
from libs.mcconversion import dualbyte


class Command(BaseCommand):
    help = "Get current MC configuration."

    def handle(self, *args, **options):
        setter = UartGetState(serial_class=PatchedSerial)
        reply, current_hues = setter.command()

        garbage, _bytes = current_hues.split(b'SD')
        _bytes = _bytes.rstrip(b'DS')
        hues = []

        loaded_state = int(_bytes[0])

        intensity = per_one_2byte.reverse(_bytes[1:3])
        fnc_count = dualbyte.reverse(_bytes[3:5])
        dim_delay = dualbyte.reverse(_bytes[5:7])

        self.stdout.write(f'load state: {loaded_state}', ending=os.linesep)
        self.stdout.write(f'{intensity=}; {fnc_count=}; {dim_delay=}',
                          ending=os.linesep)

        self.stdout.write('Current hues:', ending=os.linesep)
        for idx in range(8):
            _b = _bytes[7 + idx * 4:7 + (idx + 1) * 4]
            hues.append(full_float.reverse(_b))

        self.stdout.write(f'{hues}', ending=os.linesep)
