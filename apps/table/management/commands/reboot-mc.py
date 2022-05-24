import os
import sys
import time

from django.core.management import BaseCommand

from tablehost.uart import UartReboot
from tablehost.uart import UartError


class Command(BaseCommand):
    help = 'Reboot the configured uC.'

    def add_arguments(self, parser):
        parser.add_argument('-d', '--delay', type=float, default=2.0)

    def handle(self, *args, delay=None, **options):
        time.sleep(delay)
        rebooter = UartReboot()
        try:
            rebooter.command()
        except UartError as exc:
            self.stderr.write(f'{exc=}', ending=os.linesep)
            sys.exit(2)
