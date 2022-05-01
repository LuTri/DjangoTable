import subprocess
import uuid
import sys

from django.core.management import BaseCommand
from django.conf import settings

from django.utils.functional import lazy


def lazy_attr(self, attr_name):
    return lazy(getattr, str)(self, attr_name, None)


def lazy_concat(a, b):
    return lazy(''.join, str)([str(a), str(b)])


class Command(BaseCommand):
    help = "Push a copy of the current local source code to GIT_REMOTE_NAME."

    _opts = []

    def add_arguments(self, parser):
        parser.add_argument(
            '-r', '--remote',
            default=settings.GIT_REMOTE_NAME,
        )

        parser.add_argument(
            '-t', '--retries',
            default=1,
        )

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
        self.remote = None
        self.__tmp_branch = None

        self._opts = [
            ('current_commit', ['git', 'rev-parse', 'HEAD']),
            (None, ['git', 'add', '.']),
            (None, ['git', 'commit', '-m', '"Pushed from sheepfold"']),
            (None, ['git', 'push', '-f', lazy_attr(self, 'remote'), lazy_concat('HEAD:', lazy_attr(self, 'tmp_branch'))]),
            (None, ['git', 'reset', lazy_attr(self, 'current_commit')]),
        ]

    @property
    def tmp_branch(self):
        if self.__tmp_branch is None:
            self.__tmp_branch = settings.GIT_REMOTE_ACTIVE_BRANCH
            self.__tmp_branch = self.__tmp_branch or uuid.uuid4().hex
        return self.__tmp_branch

    def handle(self, *args, verbosity=None, retries=None, remote=None, **options):
        self.verbose = verbosity is not None and verbosity != 0
        self.retries = retries
        self.remote = remote

        cmd_status = 0

        for target_attr, opts in self._opts:
            status, value = self._run_shell_command(opts)
            if target_attr is not None:
                setattr(self, target_attr, value)
            cmd_status += status
        sys.exit(cmd_status)
