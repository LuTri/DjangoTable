from django.core.management import BaseCommand
from django.conf import settings

from libs.remoteObj import TcpWrapped
from libs.remoteObj import TcpServer

import importlib


class Command(BaseCommand):
    help = TcpWrapped.help
    epilog = TcpWrapped.epilog

    def create_parser(self, prog_name, subcommand, **kwargs):
        return super().create_parser(prog_name, subcommand, epilog=self.epilog,
                                     **kwargs)

    def add_arguments(self, parser):
        module, klass = settings.LOCAL_SERIAL_CLASS.rsplit('.', 1)
        module = importlib.import_module(module)
        klass = getattr(module, klass)
        TcpWrapped.add_arguments(parser, default_class=klass)

    def handle(self, *args, **options):
        obj = TcpServer(**options)
        obj()
