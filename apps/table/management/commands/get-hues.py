import os

from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartGetState

from libs.mcconversion import full_float


class Command(BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def add_arguments(self, parser):
        parser.add_argument('hues', metavar='HUES', type=float, nargs='*')

    def handle(self, *args, hues=(), **options):
        setter = UartGetState(serial_class=PatchedSerial)
        reply, current_hues = setter.command(*hues)

        garbage, _bytes = current_hues.split(b'SD')
        _bytes = _bytes.rstrip(b'DS')
        hues = []

        self.stdout.write('Current hues:', ending=os.linesep)
        for idx in range(8):
            _b = _bytes[idx * 4:(idx + 1) * 4]
            hues.append(full_float.reverse(_b))

        self.stdout.write(f'{hues}', ending=os.linesep)
