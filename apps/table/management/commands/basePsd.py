import importlib
import pickle
import numpy as np

from django.core.management import BaseCommand
from django.conf import settings
from libs.vbanRelay import PCMData

from libs.spectral import SpectralAudioBar


class Command(BaseCommand):
    help = "Generate and interactive PSD plotter from pickled PCM data."

    def add_arguments(self, parser):
        parser.add_argument('-d', '--data-file', default='PcmStreams.tps')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        module, klass = settings.VBAN_PRESENTER_CLASS.rsplit('.', 1)
        module = importlib.import_module(module)
        self.presenter_class = getattr(module, klass)

        self.live = False
        self._verbose = True

        self._pcm_data = None
        self.window_fnc = settings.VBAN_WELCH_WINDOW
        self.frame_idx = 0
        self.n_samples_processed = settings.VBAN_SAMPLES_PROCESSED
        self.presented_freq_range = list(settings.PRESENTER_FREQUENCY_RANGE)
        self.fft_impl = 'scipy.periodigram'

        self.__frames_of_n_samples = None
        self.__combined_frames = None

        self.needs_setup = True
        self.plotter = None
        self.audi_bar = None

    def do_setup(self, data_file=None):
        self.needs_setup = False

        if data_file is not None:
            with open(data_file, 'rb') as fp:
                self._pcm_data = pickle.load(fp)

        presenter_kwargs = {}
        if settings.DO_MATPLOT:
            presenter_kwargs.update({
                'live': self.live,
                'n_frames': len(self.combined_frames),
                'min_freq': self.presented_freq_range[0],
                'max_freq': self.presented_freq_range[1],
                'frame_update_fnc': self.get_frame_updater(),
                'window_update_fnc': self.get_window_updater(),
                'modulator_cb': self.get_modulator_update_cb(),
                'samples_update_fnc': self.get_samples_updater(),
                'min_range_updater': self.get_range_updater(for_min=True),
                'max_range_updater': self.get_range_updater(for_min=False),
                'fft_impl_updater': self.get_impl_updater(),
            })

        self.plotter = self.presenter_class(
            os_new_min=settings.STL_FFT_SCALER_NEW_MIN,
            os_new_max=settings.STL_FFT_SCALER_NEW_MAX,
            os_old_min=settings.STL_FFT_SCALER_OLD_MIN,
            os_old_max=settings.STL_FFT_SCALER_OLD_MAX,
            **presenter_kwargs,
        )
        self.audi_bar = SpectralAudioBar(verbose=self._verbose)

    @property
    def combined_frames(self):
        if self.__combined_frames is None or self.n_samples_processed != self.__frames_of_n_samples:
            t_data = None
            self.__combined_frames = []

            for item in self._pcm_data:
                _reshaped = np.reshape(item.data,
                                       (item.data.shape[0] * item.data.shape[1],),
                                       'C')

                if t_data is None:
                    t_data = np.copy(_reshaped)
                else:
                    t_data = np.concatenate([t_data, _reshaped])

                if len(t_data) / item.channels >= self.n_samples_processed:
                    self.__combined_frames.append(PCMData(
                        t_data,
                        item.channels,
                        item.sample_rate,
                        int(len(t_data) / item.channels),
                        item.frame_counter
                    ))
                    t_data = None
            self.__frames_of_n_samples = self.n_samples_processed
        return self.__combined_frames

    def get_range_updater(self, for_min=False):
        def updater(value):
            if for_min:
                self.presented_freq_range[0] = value
            else:
                self.presented_freq_range[1] = value

            self.present_frame(self.frame_idx)
        return updater

    def get_samples_updater(self):
        def updater(value):
            self.n_samples_processed = value
            self.frame_idx = 0
            self.present_frame(0)
            return len(self.combined_frames)
        return updater

    def get_modulator_update_cb(self):
        def cb():
            self.present_frame(self.frame_idx)
        return cb

    def get_frame_updater(self):
        def updater(frame_idx):
            self.frame_idx = frame_idx
            self.present_frame(frame_idx)

        return updater

    def get_window_updater(self):
        def updater(fnc_name):
            self.window_fnc = fnc_name
            self.present_frame(self.frame_idx)
        return updater

    def get_impl_updater(self):
        def updater(fft_impl):
            self.fft_impl = fft_impl
            self.present_frame(self.frame_idx)
        return updater

    def present_frame(self, idx):
        self.plotter.handle(
            *self.audi_bar(self.combined_frames[idx],
                          window_fnc=self.window_fnc,
                          mean_channels=False,
                          presented_range=self.presented_freq_range,
                          fft_impl=self.fft_impl)
        )

    def handle(self, *args, data_file=None, **options):
        if self.needs_setup:
            self.do_setup(data_file)
        self.present_frame(0)
