from django.core.management.base import BaseCommand, CommandError
import mpd

class Command(BaseCommand):
	args = ""
	help = "Reload the hosts MPD-Library"

	def handle(self, *args, **options):
		client = mpd.MPDClient(use_unicode=True)
		client.connect("localhost",6600)
		client.update()
