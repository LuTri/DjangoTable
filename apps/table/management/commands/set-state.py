import os

from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartSetState
from tablehost.uart import UartGetState
from libs.mcState import StateStruct


class Command(BaseCommand):
    IGNORE_STATES = ['load_status', 'checksum']
    help = "Set sound-to-light colors."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        statestruct = StateStruct()
        getter = UartGetState(serial_class=PatchedSerial)
        reply, self.initials = getter.command()
        getter.connection.close()

    def add_arguments(self, parser):
        for name, value in self.initials.items():
            if name in self.IGNORE_STATES:
                continue
            parser.add_argument(f'--{name.replace("_", "-")}', default=value, type=type(value))

    def handle(self, *args, **options):
        setter = UartSetState(serial_class=PatchedSerial)
        set_kwargs = {n:v for n, v in options.items() if n in self.initials.keys()}

        for k, v in set_kwargs.items():
            self.stdout.write(f'{k}: {v}', ending=os.linesep)

        setter.command(**set_kwargs)
        self.stdout.write(f'bytes written: {setter.uart_out}', ending=os.linesep)
        self.stdout.write(f'byte 8, 9: {setter.uart_out[8]}, {setter.uart_out[9]}', ending=os.linesep)

