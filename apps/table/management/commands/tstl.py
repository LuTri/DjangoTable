from django.core.management import BaseCommand
from libs.vbanRelay import VBANCollector

from libs.spectral import SpectralAudioBar
from tablehost.uart import SoundToLight
from tablehost.uart import PatchedSerial


class Command(VBANCollector, BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def handle(self, *args, slice_size=None, overlap=None, **options):
        self.run()

        slave = SoundToLight(serial_class=PatchedSerial)
        handler = SpectralAudioBar(verbose=True)

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
                    frequency_domain.write(slave)
            except KeyboardInterrupt:
                break

        self.quit()
