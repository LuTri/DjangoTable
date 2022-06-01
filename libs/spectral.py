import numpy as np
import logging
import time

from django.conf import settings

from scipy import signal


class SpectralRepresenter:
    def __init__(self, frequency_domain_data, frequencies, *args, **kwargs):
        self._frequencies = frequencies
        self._data = np.copy(frequency_domain_data)

    def as_simple_plot(self, plt_ax):
        pass


def raise_90(data):
    return data + 90 + 35


def log2_modulator(offset=6, end_val=3):
    def modulate(data):
        mod = np.log2(np.linspace(1, end_val, num=np.max(data.shape) + offset, endpoint=True))
        print(f'{mod=}')
        return data * (mod[offset] / np.max(mod))
    return modulate


def _linear(start_y=.8, end_y=1.8, b=0):

    def mod(data):
        factors = np.linspace(start_y, end_y, num=len(data), endpoint=False)
        return data * factors + 0
    return mod


# def linear_modulator(data):
#     m = (settings.PRESENTER_FREQUENCY_RANGE[1] - settings.PRESENTER_FREQUENCY_RANGE[0]) / 14
#
#     factors = np.linspace(0, len(data), num=len(data), endpoint=False, dtype='int') * -m
#
#     return data * factors

linear_modulator = _linear()


LOG2MOD = log2_modulator(offset=1, end_val=2048)

CURRENT_RANGE = {
    'min': None,
    'max': None,
}


class SpectralBarRepresenter(SpectralRepresenter):
    N_PACKS = 112

    DEFAULT_N_BARS = 14

    def __init__(self, *args, frequency_range=None, n_bars=None, **kwargs):
        super().__init__(*args, **kwargs)

        # property necessities

        self.__n_bars = n_bars
        self.__frequency_range = frequency_range
        self.__bar_frequency_values = None
        self.__bar_width = None
        self.__sampled_frequencies = self._frequencies
        self.__cover_indices = None
        self.__sample_distances = None

    @property
    def n_bars(self):
        if self.__n_bars is None:
            self.__n_bars = self.DEFAULT_N_BARS
            # Reset depending properties
            self.__bar_frequency_values = None
        return self.__n_bars

    @n_bars.setter
    def n_bars(self, val):
        if self.__n_bars != val:
            # Reset depending properties
            self.__bar_frequency_values = None
        self.__n_bars = val

    @property
    def frequency_range(self):
        if self.__frequency_range is None:
            self.__frequency_range = settings.PRESENTER_FREQUENCY_RANGE
            # Reset depending properties
            self.__bar_frequency_values = None
        return self.__frequency_range

    @frequency_range.setter
    def frequency_range(self, val):
        if self.__frequency_range is None or val != self.__frequency_range:
            # Reset depending properties
            self.__bar_frequency_values = None
            self.__frequency_range = val

    @property
    def sampled_frequencies(self):
        if self.__sampled_frequencies is None:
            raise ValueError('No samples received yet!')
        return self.__sampled_frequencies

    @sampled_frequencies.setter
    def sampled_frequencies(self, value):
        was_none = self.__sampled_frequencies is None
        if was_none or not np.all(np.equal(value, self.__sampled_frequencies)):
            # Reset depending properties
            self.__cover_indices = None
        self.__sampled_frequencies = value

    @property
    def sample_step(self):
        return self.sampled_frequencies[1] - self.sampled_frequencies[0]

    @property
    def bar_width(self):
        if self.bar_frequencies is not None:
            return self.__bar_width
        return None

    @bar_width.setter
    def bar_width(self, value):
        self.__bar_width = value

    @property
    def bar_frequencies(self):
        if self.__bar_frequency_values is None:
            bf, width = np.linspace(*self.frequency_range, num=self.n_bars,
                                    endpoint=False, retstep=True)
            self.bar_width = width
            self.__bar_frequency_values = bf + (width / 2)
            # Reset depending properties
            self.__cover_indices = None
            self.__sample_distances = None
        return self.__bar_frequency_values

    @property
    def cover_indices(self):
        if self.__cover_indices is None:
            _left_nearest = np.searchsorted(
                self.sampled_frequencies,
                self.bar_frequencies - (self.bar_width / 2)
            ) - 1
            _right_nearest = np.searchsorted(
                self.sampled_frequencies,
                self.bar_frequencies + (self.bar_width / 2)
            )

            self.__cover_indices = np.swapaxes(
                np.array([_left_nearest, _right_nearest]), 0, 1
            )
            # Reset depending properties
            self.__sample_distances = None
        return self.__cover_indices

    @property
    def sample_distances(self):
        if self.__sample_distances is None:
            self.__sample_distances = [
                np.absolute(np.array(
                    self._frequencies[slice(*self.cover_indices[idx])]
                ) - bar) for idx, bar in enumerate(self.bar_frequencies)
            ]
        return self.__sample_distances

    def write(self, writer):
        #print(f'{self.sample_distances=}')
        #print(f'{self.cover_indices=}')
        #print(f'{self.bar_frequencies=}')
        #print(f'{self.bar_width=}')

        #ref_data = [self._data, self._data]
        #ref_data = np.array(ref_data)

        #ref_data = raise_90(ref_data)
        #ref_data = LOG2MOD(ref_data)

        #ref_data = ref_data / 100

        #print(f'{ref_data=}')

        _min = 0
        _max = 130

        _bar_sections = []

        #total_energy = self._data[0]
        total_energy = 20

        _ref_max = 0

        for bar in range(self.n_bars):
            _data = self._data[slice(*self.cover_indices[bar])]
            #print(f'Data for {bar}: {_data}')
            #
            # _avg_data = [
            #     np.mean(_data[0]),
            #     np.mean(_data[1:-1].flatten()),
            #     np.mean(_data[-1]),
            # ]
            #
            #full_weighted_avg = np.average(self._data[slice(*self.cover_indices[bar])],
            #                               weights=self.sample_distances[bar])
            rss = np.sqrt(np.sum(self._data[slice(*self.cover_indices[bar])] ** 2))
            # items = sorted([np.max(_data),
            #                 full_weighted_avg,
            #                 np.min(_data)])

            tell = False
            #print(f'Items for {bar}: {items}')
            #val = np.max(np.array(items))
            val = (rss / total_energy) * 0xFFFF
            _ref_max = max([_ref_max, val])

            val_limits = settings.PRESENTER_VALUE_LIMITS
            abs_val_range = val_limits[1] + val_limits[0]

            scaler = abs_val_range / val_limits[1]
            scaled = val * scaler - val_limits[0]

            val = scaled

            #val = val * scaler - val_limits[0]

            #val = np.max(_data)
            #print(f'Val for {bar}: {val=}')
            if CURRENT_RANGE['min'] is None or val < CURRENT_RANGE['min']:
                CURRENT_RANGE['min'] = val
                tell = True
            if CURRENT_RANGE['max'] is None or val > CURRENT_RANGE['max']:
                CURRENT_RANGE['max'] = val
                tell = True

            if tell:
                print(f'NEW RANGE: {CURRENT_RANGE}')

            # if 0 < val / _max * 0xFFFF < 0xffff:
            #     _bar_sections.append(val / _max * 0xFFFF)
            # elif val / _max * 0xFFFF > 0xffff:
            #     _bar_sections.append(0xffff)
            # else:
            #     _bar_sections.append(0)

            # if 0 < 0xFFFF + (val * 0xFFFF) < 0xffff:
            #     _bar_sections.append(0xFFFF + (val * 0xFFFF))
            # elif 0xFFFF + (val * 0xFFFF) > 0xffff:
            #     _bar_sections.append(0xffff)
            # else:
            #     _bar_sections.append(0)

            _bar_sections.append(val)
            #print(f'int16 val for {bar}: {_bar_sections[-1]}')

            #refs = np.linspace(_min, _max, 8, endpoint=True, retstep=False)
            #for y in range(8):
            #    h = 1/14 * y

            #    if any([c >= refs[y] for c in items]):
            #        m = np.average(np.array(items))
            #        r, g, b = colorsys.hsv_to_rgb(h, 1, (m / _max) * .5)
            #        preset[coord_to_snakish(bar, 7 - y)] = (int(g*0xff) & 0xff,
            #                                                int(r*0xff) & 0xff,
            #                                                int(b*0xff) & 0xff)

        #raw = []
        #for key in sorted(preset.keys()):
        #    for item in preset[key]:
        #        raw.append(item)
        #print(f'N_vals: {len(_bar_sections)}')

        _bar_sections = np.array(_bar_sections)
        _bar_sections[~np.isfinite(_bar_sections)] = 0

        _bar_sections = _bar_sections + np.abs(np.max(_bar_sections) - _ref_max)

        cmd_kwargs = {f'val_{x}': int(max([min([value, 0xFFFF]), 0])) for x, value in enumerate(_bar_sections)}

        writer.command(max_intensity=settings.STL_MAX_INTENSITY,
                       dim_steps=settings.STL_DIM_STEPS,
                       dim_delay=settings.STL_DIM_DELAY,
                       **cmd_kwargs)


class SpectralAudioBar:
    MAX_BYTE_BUFFER = 16384
    FREE_BYTES = 1024

    @staticmethod
    def byte_size(obj):
        try:
            return np.prod(obj.shape) * obj.itemsize
        except AttributeError:
            return len(obj)

    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.verbose = verbose
        self.logger = logging.getLogger('uart_threads')

        self.parsed_frames = 0
        self.last_analyzed_frame = 0
        self.last_handled_frame = 0

        self.previous_plot_time = None
        self.running = True
        self.last_frame = None

        self.analyzing = True

        self._frequencies = None
        self._last_pcm = None
        self._i_pcm_start = 0

        # NEW ATTRIBUTES

        self.requested_slice = 0
        self.requested_overlap = 0

        self._seek = 0
        self._seek_out = 0

        self.frame_buffer = np.array([], dtype=np.dtype('int16'))
        self.dbfs_buffer = []
        self.frequencies_buffer = []

    def fft_frequency_domain(self, sample_rate, window=None,
                             ref=32768):
        """
        https://dsp.stackexchange.com/a/32080
        
        Calculate spectrum in dB scale
        """
        num = len(self.frame_buffer) # Length of input sequence

        if window is None:
            window = np.ones(num)
        if num != len(window):
            raise ValueError('Signal and window must be of the same length')
        signal_data = self.frame_buffer * window

        # Calculate real FFT and frequency vector
        spectrum = np.fft.rfft(signal_data)
        frequencies = np.fft.rfftfreq(num) * sample_rate

        # Scale the magnitude of FFT by window and factor of 2,
        # because we are using half of FFT spectrum.
        s_magnitude = np.abs(spectrum) * 2 / np.sum(window)

        # Fix inifinite or 0 magnitudes
        s_magnitude[~np.isfinite(s_magnitude)] = 1.0
        s_magnitude[s_magnitude == 0.] = 1.0

        # Convert to dBFS
        return frequencies, 20 * np.log10(s_magnitude / ref)

    def process_data(self, sample_rate, fft_size=None):
        timing = time.time()
        if fft_size is None:
            fft_size = len(self.frame_buffer)

        frequencies, Pxx_den = signal.welch(
            self.frame_buffer,
            sample_rate,
            window='boxcar',
            nperseg=settings.VBAN_SAMPLES_PROCESSED / 2,
            noverlap=settings.VBAN_SAMPLES_PROCESSED / 4,
            nfft=settings.VBAN_SAMPLES_PROCESSED,
            scaling='spectrum',
            detrend='constant',
            return_onesided=True,
            average='mean'
        )

        Pxx_den[Pxx_den == 0.] = 1.
        Pxx_den = np.log10(Pxx_den)
        Pxx_den[~np.isfinite(Pxx_den)] = 0

        representer = SpectralBarRepresenter(frequency_domain_data=Pxx_den,
                                             frequencies=frequencies)
        if self.verbose:
            time_passed = time.time() - timing
            if time_passed == 0:
                time_passed = 1 / sample_rate
            time_of_frame = 1 / sample_rate
            self.logger.info(
                f'Time for fft-{fft_size}: {time_passed:.5f}; '
                f'time of frame: {time_of_frame:.5f}; '
                f'Frames per fft: {(1 / time_of_frame) / (1 / time_passed):.5f}'
            )
        return representer

    def __call__(self, signal_object, data_channel=0, fft_size=None):
        if self.byte_size(self.frame_buffer) > self.MAX_BYTE_BUFFER:

            self.frame_buffer = self.frame_buffer[self.byte_size(signal_object.data[:, data_channel]):]
            self._seek -= self.byte_size(signal_object.data[:, data_channel])

        self.frame_buffer = signal_object.data[:, data_channel]

        return self.process_data(
            signal_object.sample_rate,
            fft_size or signal_object.n_samples,
        )
