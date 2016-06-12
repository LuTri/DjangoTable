from django.core.management import BaseCommand
from django.core.management import CommandError

import commands

class RetryException(Exception):
	pass

class Command(BaseCommand):
	max_cmd_tries = 10

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
				if status != 0:
					raise RetryException(output)
				done = True
				break

			except RetryException:
				tries += 1

		return done, output

	def clean(self):
		return self._run_shell_command(['make', 'clean'])

	def make(self):
		done, output = self.clean()
		if done:
			return self._run_shell_command(['make'])
		return done, output

	def deploy(self):
		done, output = self.make()
		if done:
			return self._run_shell_command(['make', 'program'])
		return done, output

	def handle(self, *args, **options):
		if len(args) < 1:
			action = "deploy"
		else:
			action = args[0]
		try:
			done, output = getattr(self, action)()
			if done:
				print "Successfully finished command."
			else:
				raise CommandError(output)
		except AttributeError:
			raise CommandError("No such action \"%s\"" % action)
