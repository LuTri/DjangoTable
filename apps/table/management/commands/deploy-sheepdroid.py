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

        parser.add_argument(
            '-e', '--extra-ssh-cmd',
            required=False,
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
        self._resets = []

        self._tmp_heads = []

        self._opts = [
            ('current_commit', ['git', 'rev-parse', 'HEAD']),
            (None, ['git', 'add', '.']),
            (None, ['git', 'commit', '-m', '"Pushed from sheepfold"']),
            (None, ['git', 'push', '-f', lazy_attr(self, 'remote'), lazy_concat('HEAD:', lazy_attr(self, 'tmp_branch'))]),
            (None, ['git', 'reset', lazy_attr(self, 'current_commit')]),
        ]

    def _delete_remote_heads(self):
        for remote, ref_spec in self._tmp_heads:
            if self.verbose:
                self.stdout.write(f'Deleting {ref_spec} from {remote}',
                                  ending=os.linesep)
            remote.push(('-d', ref_spec))

    def _reset_heads(self):
        for commit in self._resets:
            if self.verbose:
                self.stdout.write(f'resetting {commit.repo} to {commit}',
                                  ending=os.linesep)
            commit.repo.git.reset(commit.hexsha)

    def _handle_repo(self, repo, remote_head=None):
        if self.verbose:
            self.stdout.write(f'Entering {repo}.', ending=os.linesep)
        _resets = []
        if repo.is_dirty():
            for module in repo.iter_submodules():
                try:
                    self._handle_repo(Repo(module.abspath))
                except git.exc.NoSuchPathError as exc:
                    if self.verbose:
                        self.stdout.write(
                            f'FATAL {module}: {exc}', ending=os.linesep
                        )

            if self.verbose:
                self.stdout.write(f'pushing {repo}.', ending=os.linesep)
            self._resets.append(repo.head.commit)

            _delete_remote = remote_head is None
            ref_spec = remote_head or f'T{uuid.uuid4().hex}'[:10]

            repo.git.add(repo.git.working_dir)
            repo.git.commit('-m', 'LATEST FROM DEV HOST')

            if self.verbose:
                self.stdout.write(f'{repo}: -> {ref_spec}', ending=os.linesep)
            _remote = repo.remotes.origin
            _remote.push(('-f', f'HEAD:{ref_spec}'))
            if _delete_remote:
                self._tmp_heads.append((_remote, ref_spec))

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
            self._handle_repo(self._local_repo, settings.GIT_REMOTE_ACTIVE_BRANCH)

            _url, _path = self._local_repo.remotes.origin.url.split(':')
            status, value = self._run_shell_command(
                ['ssh', '-i', 'sheepdroidDevDevploy', _url,
                 f'"cd {_path}; git submodule update --init --recursive; {extra_ssh_cmd or ""}"']
            )
            self.stdout.write(value)
        finally:
            self._delete_remote_heads()
            self._reset_heads()

        sys.exit(cmd_status)
