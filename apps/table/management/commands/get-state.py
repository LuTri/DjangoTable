import os

from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartGetState

from libs.mcState import StateStruct


class Command(BaseCommand):
    help = "Get current MC configuration."

    def handle(self, *args, **options):
        setter = UartGetState(serial_class=PatchedSerial)
        reply, data = setter.command()

        for k, v in data.items():
            self.stdout.write(f'{k}: {v}', ending=os.linesep)
