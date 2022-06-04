from django.core.management import BaseCommand
from tablehost.uart import PatchedSerial
from tablehost.uart import UartSetState


class Command(BaseCommand):
    help = "Set sound-to-light colors."

    def add_arguments(self, parser):
        parser.add_argument('-i', '--intensity', default=.7, type=float)
        parser.add_argument('-f', '--fnc-count', default=60000, type=int)
        parser.add_argument('-d', '--dim-count', default=1000, type=int)
        parser.add_argument('hues', metavar='HUES', type=float, nargs='*')

    def handle(self, *args, intensity=None, fnc_count=None, dim_count=None, hues=(), **options):
        setter = UartSetState(serial_class=PatchedSerial)
        setter.command(*hues, intensity=intensity, fnc_count=fnc_count, dim_count=dim_count)
