import os
import subprocess
import uuid
import sys

import git.exc
from django.core.management import BaseCommand
from django.conf import settings

from django.utils.functional import lazy
from git import Repo


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

        parser.add_argument(
            '-u', '--update-remote',
            action='store_false',
            default=True
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

        self._local_repo = Repo(os.curdir)

        self._opts = [
            ('current_commit', ['git', 'rev-parse', 'HEAD']),
            (None, ['git', 'add', '.']),
            (None, ['git', 'commit', '-m', '"Pushed from sheepfold"']),
            (None, ['git', 'push', '-f', lazy_attr(self, 'remote'),
                    lazy_concat('HEAD:', lazy_attr(self, 'tmp_branch'))]),
            (None, ['git', 'reset', lazy_attr(self, 'current_commit')]),
        ]

    def _handle_repo(self, repo):
        if self.verbose:
            self.stdout.write(f'Entering {repo}.', ending=os.linesep)

        if repo.is_dirty():
            for module in repo.iter_submodules():
                try:
                    self._handle_repo(Repo(module.abspath))
                except git.exc.NoSuchPathError as exc:
                    self.stderr.write(
                        f'FATAL {module}: {exc}', ending=os.linesep
                    )

            if self.verbose:
                self.stdout.write(f'pushing {repo}.', ending=os.linesep)

            repo.git.add(repo.git.working_dir)
            repo.git.commit('-m', 'INDEV auto deploy')

            if self.verbose:
                self.stdout.write(f'{repo}: -> {self.tmp_branch}',
                                  ending=os.linesep)
            _remote = repo.remotes.origin
            _remote.push(('-f', f'HEAD:{self.tmp_branch}'))

    @property
    def tmp_branch(self):
        if self.__tmp_branch is None:
            self.__tmp_branch = settings.GIT_REMOTE_ACTIVE_BRANCH
            self.__tmp_branch = self.__tmp_branch or uuid.uuid4().hex
        return self.__tmp_branch

    def handle(self, *args, verbosity=None, retries=None, remote=None,
               extra_ssh_cmd=None, **options):
        self.verbose = verbosity is not None and verbosity > 1
        self.retries = retries
        self.remote = remote

        cmd_status = 0

        try:
            self._handle_repo(self._local_repo)

            _url, _path = self._local_repo.remotes.origin.url.split(':')
            status, value = self._run_shell_command(
                ['ssh', '-i', 'sheepdroidDevDevploy', settings.ODROID_HOST_NAME,
                 (f'"cd {_path}; git fetch origin && git reset --hard '
                  f'origin/{self.tmp_branch} && git submodule update --init '
                  f'--recursive"')]
            )
            self.stdout.write(value)
        except Exception as exc:
            self.stderr.write(f'Error: {exc}', ending=os.linesep)
            raise
        finally:
            sys.exit(cmd_status)
