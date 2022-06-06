import os

from django.utils import autoreload
from django.conf import settings
from libs.vbanRelay import VBANCollector

from .basePsd import Command as PsdCommand


class Command(VBANCollector, PsdCommand):
    help = "Generate and interactive PSD plotter for streamed PCM data."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.live = True
        self._verbose = False

        self._kwargs.update({'required_samples': self.n_samples_processed})

        self.__live_frame = None
        self.last_frame = 0

    def add_arguments(self, parser):
        parser.add_argument('-a', '--auto-reload', action='store_true',
                            default=False)

        parser.add_argument('-c', '--continuous', action='store_true',
                            default=False)

    @property
    def combined_frames(self):
        return self.__live_frame

    @combined_frames.setter
    def combined_frames(self, value):
        self.__live_frame = value

    def run_once(self):
        frames = self.last_n_pcm(1)
        if frames and frames[-1].frame_counter != self.last_frame:
            _last_frame = frames[-1].frame_counter
            skipped = _last_frame - self.last_frame
            if skipped > 1 and self._verbose:
                self.stdout.write(f'Frames skipped: {skipped}')
            self.last_frame = _last_frame

            self.combined_frames = frames
            if self.needs_setup:
                self.do_setup()

            self.present_frame(-1)

    def inner_run(self, continuous=False, **options):
        self.stderr.write(f'Config:', ending=os.linesep)
        self.stderr.write(f'{settings.PRESENTER_VALUE_LIMITS=}', ending=os.linesep)
        self.stderr.write(f'{settings.PRESENTER_FREQUENCY_RANGE=}', ending=os.linesep)
        self.stderr.write(f'{settings.VBAN_SAMPLES_PROCESSED=}', ending=os.linesep)

        self.run()

        try:
            if continuous:
                while True:
                    self.run_once()
            else:
                self.do_setup()
                self.plotter.run()
        finally:
            self.quit()

    def handle(self, *args, auto_reload=None, **options):
        if auto_reload:
            autoreload.run_with_reloader(self.inner_run, **options)
        else:
            self.inner_run(**options)

    def get_impl_updater(self):
        def updater(fft_impl):
            self.fft_impl = fft_impl
        return updater

    def get_modulator_update_cb(self):
        def cb():
            pass
        return cb

    def get_frame_updater(self):
        return None

    def get_window_updater(self):
        def updater(fnc_name):
            self.window_fnc = fnc_name
        return updater

    def get_samples_updater(self):
        def updater(value):
            self.n_samples_processed = value
            self.frame_idx = 0
            self.quit()
            self._kwargs.update({'required_samples': self.n_samples_processed})
            self.run()

        return updater

    def get_range_updater(self, for_min=False):
        def updater(value):
            if for_min:
                self.presented_freq_range[0] = value
            else:
                self.presented_freq_range[1] = value
        return updater
