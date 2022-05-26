import importlib

from django.core.management import BaseCommand
from django.conf import settings
from libs.vbanRelay import VBANCollector

from libs.spectral import SpectralAudioBar


class Command(VBANCollector, BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        module, klass = settings.VBAN_PRESENTER_CLASS.rsplit('.', 1)
        module = importlib.import_module(module)
        self._presenter_class = getattr(module, klass)

    def handle(self, *args, slice_size=None, overlap=None, **options):
        self.run()

        slave = self._presenter_class()
        handler = SpectralAudioBar(verbose=False)

        last_frame = 0
        while True:
            try:
                frames = self.last_n_pcm(1)
                if frames and frames[-1].frame_counter != last_frame:
                    _last_frame = frames[-1].frame_counter
                    #skipped = _last_frame - last_frame
                    #if skipped > 1:
                    #    self.stdout.write(f'Frames skipped: {skipped}')
                    last_frame = _last_frame
                    frequency_domain = handler(frames[-1])
                    slave.handle(frequency_domain)
                    #frequency_domain.write(slave)
            except KeyboardInterrupt:
                break

        self.quit()