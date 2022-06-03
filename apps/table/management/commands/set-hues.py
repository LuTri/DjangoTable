from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartSetState


class Command(BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def add_arguments(self, parser):
        parser.add_argument('hues', metavar='HUES', type=float, nargs='*')

    def handle(self, *args, hues=(), **options):
        setter = UartSetState(serial_class=PatchedSerial)
        setter.command(*hues)
