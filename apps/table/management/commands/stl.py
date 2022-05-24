import sys
import time

from django.core.management import BaseCommand
from libs.vbanRelay import VBANCollector
from django.conf import settings

from tablehost.uart import UartSlave
from libs.spectral import SpectralAudioBar


class Command(VBANCollector, BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def add_arguments(self, parser):
        parser.add_argument('-p', '--port', required=False)

    def handle(self, *args, port=None, **options):
        avg_time = time.time()
        cmd_counter = 0
        analyzer = SpectralAudioBar()

        slave = UartSlave()

        frame_counter = 0
        t_counter = None
        rc = 0
        try:
            self.run()
            while True:
                cmd_counter += 1
                avg_time = (avg_time + time.time()) / 2
                t_counter = self.get_frame_counter()
                if t_counter == frame_counter:
                    continue

                frame_counter = t_counter
                frames = self.last_n_pcm(1)
                if frames:
                    if analyzer(frames[-1]):
                        fd = analyzer.fetch()
                        fd.write(slave)
                else:
                    print("NO FRAME :(")
                if cmd_counter % 1000 == 0:
                    print(f'{cmd_counter} frames so far, avg time: {avg_time}.')

        except KeyboardInterrupt:
            self.stdout.write('Good bye!', ending='\n')
        except Exception as exc:
            self.stderr.write(str(exc), ending='\n')
            raise
        finally:
            slave.close()
            self.quit()
        sys.exit(rc)
