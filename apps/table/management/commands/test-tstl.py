import pickle
import numpy as np

from django.core.management import BaseCommand
from django.conf import settings
from libs.vbanRelay import VBANCollector
from libs.vbanRelay import PCMData
from libs.testplotterthread import Pyplotter
from libs.testplotterthread import AVAILABLE_WINDOWS

from libs.spectral import SpectralAudioBar


class Command(VBANCollector, BaseCommand):
    help = "SOUND TO LIGHT TEST (via slaving)."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open('PcmStreams.tps', 'rb') as fp:
            pcm_data = pickle.load(fp)

        t_data = None
        self.large_data = []
        self.window_fnc = AVAILABLE_WINDOWS[0]
        self.frame_idx = 0

        for item in pcm_data:
            _reshaped = np.reshape(item.data,
                                   (item.data.shape[0] * item.data.shape[1],),
                                   'C')

            if t_data is None:
                t_data = np.copy(_reshaped)
            else:
                t_data = np.concatenate([t_data, _reshaped])

            if len(t_data) / item.channels >= settings.VBAN_SAMPLES_PROCESSED:
                self.large_data.append(PCMData(
                    t_data,
                    item.channels,
                    item.sample_rate,
                    int(len(t_data) / item.channels),
                    item.frame_counter
                ))
                t_data = None

        self.plotter = Pyplotter(ion=False, n_frames=len(self.large_data), frame_update_fnc=self.get_frame_updater())
        self.audi_bar = SpectralAudioBar()

    def get_frame_updater(self):

        def updater(frame_idx):
            self.frame_idx = frame_idx
            self.present_frame(frame_idx)

        return updater

    def get_window_update(self):

        def updater(idx):
            self.window_fnc = AVAILABLE_WINDOWS[idx]
            self.present_frame(self.frame_idx)
        return updater

    def present_frame(self, idx):
        self.plotter.handle(self.audi_bar(self.large_data[idx], window_fnc=self.window_fnc))

    def handle(self, *args, **options):
        self.present_frame(0)
