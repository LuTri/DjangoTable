import os
import subprocess
import sys

from django.core.management import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('-b', '--branch', default='indev', required=False)
        parser.add_argument('-r', '--retries', default=1, required=False, type=int)

    def _run_shell_command(self, opts):
        tries = 0
        opts = [] + opts

        status = None
        output = None

        cmd = ' '.join([str(x) for x in opts])

        if self.verbose:
            print(cmd)

        while tries < self.retries:
            status, output = subprocess.getstatusoutput(cmd)
            if self.verbose:
                print(f'{status=}, {output=}')
            if int(status) != 0:
                tries += 1
                continue
            break

        return status, output

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = None
        self.retries = None

    def handle(self, *args, branch=None, retries=None, **options):
        self.retries = retries
        cmd_status = 0

        try:
            status, value = self._run_shell_command(
                ['ssh', '-i', 'sheepdroidDevDevploy',
                 f'{settings.ODROID_HOST_USER}@{settings.ODROID_HOST_NAME}',
                 (f'"cd {settings.ODROID_PROJECT_PATH}; source bin/activate; '
                  f'cd {settings.ODROID_MAKE_PATH}; make clean && make; '
                  f'make program || make program"')]
            )
            self.stdout.write(value)
        except Exception as exc:
            self.stderr.write(f'Error: {exc}', ending=os.linesep)
            raise
        sys.exit(cmd_status)
