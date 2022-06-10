import os

from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartGetState


class Command(BaseCommand):
    help = "Get current MC configuration."

    def handle(self, *args, **options):
        setter = UartGetState(serial_class=PatchedSerial)
        reply, data = setter.command()

        self.stdout.write(f'load state: {data["load_status"]}', ending=os.linesep)
        self.stdout.write(f'intensity: {data["intensity"]}; fnc_count: {data["fnc_count"]}; dim_delay: {data["dim_delay"]}',
                          ending=os.linesep)

        self.stdout.write(f'Current hues: {data["hues"]}', ending=os.linesep)
