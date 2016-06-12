from django.core.management import BaseCommand
from django.core.management import CommandError

from optparse import make_option

import commands

class RetryException(Exception):
	pass

class Command(BaseCommand):
	max_cmd_tries = 10

	option_list = BaseCommand.option_list + (
		make_option('--head', action='store', dest='head', default='origin/master'),
	)		

	def __init__(self, *args, **kwargs):
		self.verbose = True
		super(Command, self).__init__(*args, **kwargs)

	def _run_shell_command(self, opts):
		done = False
		tries = 0
		opts = ['cd', 'AVRMusicTable', ';'] + opts
		cmd = ' '.join(opts)
		if self.verbose:
			print cmd
		while tries < 10:
			try:
				status, output = commands.getstatusoutput(cmd)
				if self.verbose:
					print output
				if status != 0:
					raise RetryException(output)
				done = True
				break

			except RetryException:
				tries += 1

		return done, output

	def fetch(self, head, **options):
		opts = ['git', 'fetch', '&&', 'git', 'reset', '--hard', head, '&&']
		opts += ['git', 'submodule', 'foreach', '--recursive', '"']
		opts += ['git', 'fetch', '&&', 'git', 'reset', '--hard', 'origin/master', '"']
		return self._run_shell_command(opts)

	def clean(self, **options):
		return self._run_shell_command(['make', 'clean'])

	def make(self, **options):
		d_fetch, ouoput = self.fetch(**options)
		d_clean, output = self.clean(**options)
		if d_fetch and d_clean:
			return self._run_shell_command(['make'])
		return done, output

	def deploy(self, **options):
		done, output = self.make(**options)
		if done:
			return self._run_shell_command(['make', 'program'])
		return done, output

	def handle(self, *args, **options):
		if len(args) < 1:
			action = "deploy"
		else:
			action = args[0]

		try:
			done, output = getattr(self, action)(**options)
			if done:
				print "Successfully finished command."
			else:
				raise CommandError(output)
		except AttributeError:
			raise CommandError("No such action \"%s\"" % action)
