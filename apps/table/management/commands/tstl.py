from django.core.management import BaseCommand
from libs.vbanRelay import VBANCollector

from libs.spectral import SpectralAudioBar
from tablehost.uart import UartSlave


class Command(VBANCollector, BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def handle(self, *args, slice_size=None, overlap=None, **options):
        self.run()

        slave = UartSlave()

        handler = SpectralAudioBar(verbose=True)
        last_frame = 0
        idx = 0
        while idx < 190000:
            frames = self.last_n_pcm(1)
            if frames and frames[-1].frame_counter != last_frame:
                _last_frame = frames[-1].frame_counter
                self.stdout.write(f'Frames skipped: {_last_frame - last_frame}')
                last_frame = _last_frame
                frequency_domain = handler(frames[-1])
                frequency_domain.write(slave)
                idx += 1
        self.quit()