import numpy as np
import logging
import time

from django.conf import settings

from scipy import signal
from copy import deepcopy
from scipy.signal.windows import get_window


def make_linear_output_scaler(new_max, new_min, old_max, old_min, as_int=True):
    def mod(data):
        t_arr = (new_max - new_min) * ((data - old_min) / (old_max - old_min)) + new_min
        if as_int:
            t_arr = t_arr.astype('int')
            t_arr[t_arr < 0] = 0
            t_arr[t_arr > new_max] = new_max
        return t_arr
    return mod


def raise_90(data):
    return data + 90 + 35


def log2_modulator(offset=6, end_val=2000):
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


LOG2MOD = log2_modulator(offset=1, end_val=2048)


CURRENT_RANGE = {
    'min': None,
    'max': None,
}

DATA_COMBINER = {
    'light_variance': lambda _data: np.average(_data, weights=1 / np.abs(_data - np.mean(_data))),
    'heavy_variance': lambda _data: np.average(_data, weights=np.abs(_data - np.mean(_data)) ** 2),
    'variance': lambda _data: np.average(_data, weights=np.abs(_data - np.mean(_data))),
    'median': lambda _data: np.median(_data),
    'mean': lambda _data: np.mean(_data),
    'max': lambda _data: np.max(_data),
    'min': lambda _data: np.min(_data),
}


class SpectralBarRepresenter:
    N_PACKS = 112
    MOD_ENDVAL = 2000
    MOD_OFFSET = 26
    MOD_AMPLIFY = 1.3

    DEFAULT_N_BARS = 14

    def __init__(self, n_bars=None, verbose=False):
        self.verbose = verbose
        # property necessities
        self.__n_bars = n_bars
        self.__frequency_range = None
        self.__bar_frequency_values = None
        self.__bar_borders = None
        self.__sampled_frequencies = None
        self.__cover_indices = None
        self.__sample_distances = None
        self.__data = None
        self.__sample_rate = None

    @property
    def sample_rate(self):
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self.__sample_rate = value

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        #mod = np.log2(np.linspace(1, self.MOD_ENDVAL, num=len(data) + self.MOD_OFFSET, endpoint=True))
        #self.__data = np.copy(data) * (mod[self.MOD_OFFSET:] / np.max(mod) * self.MOD_AMPLIFY)
        self.__data = np.copy(data)

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
            self.__bar_borders = None
        return self.__frequency_range

    @frequency_range.setter
    def frequency_range(self, val):
        if self.__frequency_range is None or val != self.__frequency_range:
            # Reset depending properties
            self.__bar_frequency_values = None
            self.__bar_borders = None
            self.__frequency_range = deepcopy(val)

    @property
    def sampled_frequencies(self):
        if self.__sampled_frequencies is None:
            raise ValueError('No samples received yet!')
        return self.__sampled_frequencies

    @sampled_frequencies.setter
    def sampled_frequencies(self, value):
        was_none = self.__sampled_frequencies is None
        d_size = was_none or (len(self.__sampled_frequencies) != len(value))
        if was_none or d_size or not np.all(np.equal(value, self.__sampled_frequencies)):
            # Reset depending properties
            self.__cover_indices = None
            self.__sample_distances = None
            self.__sampled_frequencies = np.copy(value)

    def _calc_bars(self):
        range_exp = np.log10(self.frequency_range[1]) / np.log10(10)
        if self.frequency_range[0] != 0:
            start_exp = np.log10(self.frequency_range[0]) / np.log10(10)
        else:
            start_exp = 0

        bf_log = np.linspace(start_exp, range_exp,
                             num=(self.n_bars * 2) + 1,
                             endpoint=True)
        self.__bar_frequency_values = np.power(10, bf_log[1::2])
        self.__bar_borders = np.power(10, bf_log[::2])

        # Reset depending properties
        self.__cover_indices = None
        self.__sample_distances = None

    @property
    def bar_borders(self):
        if self.__bar_borders is None:
            self._calc_bars()
        return self.__bar_borders

    @property
    def bar_frequencies(self):
        if self.__bar_frequency_values is None:
            self._calc_bars()
        return self.__bar_frequency_values

    @property
    def cover_indices(self):
        if self.__cover_indices is None:
            nearest = np.searchsorted(
                self.sampled_frequencies,
                self.bar_borders
            )
            _left_nearest = nearest[:-1]

            _right_nearest = nearest[1:]

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
                    self.sampled_frequencies[slice(*self.cover_indices[idx])]
                ) - bar) for idx, bar in enumerate(self.bar_frequencies)
            ]
        return self.__sample_distances

    def write(self, writer, method=None, o_scale_new_min=0,
              o_scale_new_max=0xFFFF, o_scale_old_min=0, o_scale_old_max=0xFFFF):

        _bar_sections = []

        out_scale = make_linear_output_scaler(o_scale_new_max, o_scale_new_min, o_scale_old_max, o_scale_old_min)

        if self.verbose:
            print(f'{self.sampled_frequencies=}')
            print(f'{self.frequency_range=}')
            print(f'{self.bar_frequencies=}')
            print(f'{self.cover_indices=}')

        for bar in range(self.n_bars):
            _data = self.data[slice(*self.cover_indices[bar])]
            _bar_sections.append(DATA_COMBINER[method](_data))

        _bar_sections = out_scale(np.array(_bar_sections))
        cmd_kwargs = {f'val_{x}': value for x, value in enumerate(_bar_sections)}

        writer.command(**cmd_kwargs)


class SpectralAudioBar:
    MAX_BYTE_BUFFER = 16384
    FREE_BYTES = 1024

    @staticmethod
    def byte_size(obj):
        try:
            return np.prod(obj.shape) * obj.itemsize
        except AttributeError:
            return len(obj)

    def __init__(self, verbose=False, representer_class=SpectralBarRepresenter,
                 *args, **kwargs):
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

        self.representer = representer_class(verbose=verbose)

    def kramer_fft(self, sample_rate, window=None):
        """
        https://mark-kramer.github.io/Case-Studies-Python/03.html

        Calculate spectrum in dB scale
        """
        num = len(self.frame_buffer)  # Length of input sequence

        if window is None:
            window = np.ones(num)
        if num != len(window):
            raise ValueError('Signal and window must be of the same length')
        signal_data = self.frame_buffer * window

        dt = 1 / sample_rate
        T = dt * num

        frequencies = np.fft.rfftfreq(num) * sample_rate
        # Calculate real FFT and frequency vector
        transformed = np.fft.rfft(signal_data - signal_data.mean())
        spectrum = 2 * dt ** 2 / T * (transformed * transformed.conj())

        spectrum = np.real(spectrum) / np.sum(window)



        # Scale the magnitude of FFT by window and factor of 2,
        # because we are using half of FFT spectrum.
        # s_magnitude = np.abs(spectrum) * 2 / np.sum(window)

        # Fix inifinite or 0 magnitudes
        # s_magnitude[~np.isfinite(s_magnitude)] = 1.0
        # s_magnitude[s_magnitude == 0.] = 1.0

        #spectrum[~np.isfinite(spectrum)] = -60
        #spectrum[spectrum == 0.0] = -60
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        spectrum = 10 * np.log10(spectrum[1:] / np.mean(spectrum))
        #spectrum[~np.isfinite(spectrum)] = -60
        #spectrum[]

        # Convert to dBFS
        return frequencies[1:], spectrum  # 10 * np.log10(s_magnitude)

    def compute_fft(self, sampling_rate, n=None, scale_amplitudes=True):
        '''Computes an FFT on signal s using numpy.fft.fft.

           Parameters:
            s (np.array): the signal
            sampling_rate (num): sampling rate
            n (integer): If n is smaller than the length of the input, the input is cropped. If n is
                larger, the input is padded with zeros. If n is not given, the length of the input signal
                is used (i.e., len(s))
            scale_amplitudes (boolean): If true, the spectrum amplitudes are scaled by 2/len(s)
        '''
        if n == None:
            n = len(self.frame_buffer)

        fft_result = np.fft.fft(self.frame_buffer, n)
        num_freq_bins = len(fft_result)
        fft_freqs = np.fft.fftfreq(num_freq_bins, d=1 / sampling_rate)
        half_freq_bins = num_freq_bins // 2

        fft_freqs = fft_freqs[:half_freq_bins]
        fft_result = fft_result[:half_freq_bins]
        fft_amplitudes = np.abs(fft_result)

        if scale_amplitudes is True:
            fft_amplitudes = 2 * fft_amplitudes / (len(self.frame_buffer))
        else:
            fft_amplitudes = 10 * np.log10(fft_amplitudes)

        return (fft_freqs, fft_amplitudes)

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
        return frequencies, (20 * np.log10(s_magnitude / ref))

    def signal_welch(self, sample_rate, window_name):
        frequencies, fdd = signal.welch(
            self.frame_buffer,
            sample_rate,
            window=window_name or settings.VBAN_WELCH_WINDOW,
            nperseg=int(len(self.frame_buffer) / settings.VBAN_WELCH_N_SEGMENT),
            noverlap=int(len(self.frame_buffer) / settings.VBAN_WELCH_OVERLAP),
            nfft=len(self.frame_buffer),
            scaling=settings.VBAN_WELCH_SCALING,
            detrend=settings.VBAN_WELCH_DETREND,
            return_onesided=True,
            average='mean'
        )
        #
        # fdd[fdd == 0.] = 1.
        fdd = 10 * np.log10(fdd[1:])
        # fdd[~np.isfinite(fdd)] = 0
        return frequencies[1:], fdd

    def signal_periodigram(self, sample_rate, window):
        frequencies, fdd = signal.periodogram(
            self.frame_buffer,
            sample_rate,
            window,
            scaling=settings.VBAN_WELCH_SCALING
        )

        fdd = 10 * np.log10(fdd[1:])
        return frequencies[1:], fdd

    def process_data(self, sample_rate, fft_size=None, window_fnc=None, fft_impl=None):
        timing = time.time()
        if fft_size is None:
            fft_size = len(self.frame_buffer)

        # frequencies, fdd = signal.welch(
        #     self.frame_buffer,
        #     sample_rate,
        #     window=window_fnc or settings.VBAN_WELCH_WINDOW,
        #     nperseg=settings.VBAN_WELCH_N_SEGMENT,
        #     noverlap=settings.VBAN_WELCH_OVERLAP,
        #     nfft=settings.VBAN_SAMPLES_PROCESSED,
        #     scaling=settings.VBAN_WELCH_SCALING,
        #     detrend=settings.VBAN_WELCH_DETREND,
        #     return_onesided=True,
        #     average='mean'
        # )
        #
        # fdd[fdd == 0.] = 1.
        # fdd = np.log10(fdd)
        # fdd[~np.isfinite(fdd)] = 0

        window = get_window(window_fnc or settings.VBAN_WELCH_WINDOW, len(self.frame_buffer))

        if fft_impl is None or fft_impl == 'scipy.periodigram':
            frequencies, fdd = self.signal_periodigram(sample_rate, window)
        elif fft_impl == 'kramer':
            frequencies, fdd = self.kramer_fft(sample_rate, window)
        elif fft_impl == 'scipy.welch':
            frequencies, fdd = self.signal_welch(sample_rate, window_fnc)
        else:
            frequencies, fdd = self.compute_fft(sample_rate, scale_amplitudes=False)


        #fdd[fdd == 0.] = .01
        #fdd = 10 * np.log10(fdd[1:])
        #Pxx_den[~np.isfinite(Pxx_den)] = 0

        #representer = SpectralBarRepresenter(frequency_range=presented_range,
        #                                     frequency_domain_data=fdd[1:],
        #                                     frequencies=frequencies[1:])

        self.representer.sampled_frequencies = frequencies
        self.representer.data = fdd
        self.representer.sample_rate = sample_rate

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
        return self.representer

    def __call__(self, signal_object, data_channel=0, mean_channels=False,
                 fft_size=None, window_fnc=None, presented_range=None, fft_impl=None):

        #dt = 1 / signal_object.sample_rate
        #t = dt * 256
        #theta = 0
        #frequency = 440
        #amplitude = 30000
        #time = np.arange(0, t, dt)

        #self.frame_buffer = np.pad(amplitude * np.sin(2 * np.pi * frequency * time + theta), (1, len(signal_object.data[:, data_channel]) - 256))
        if mean_channels:
            self.frame_buffer = signal_object.mean_channels()
        else:
            self.frame_buffer = signal_object.data[:, data_channel]

        self.representer.frequency_range = presented_range

        return self.process_data(
            signal_object.sample_rate,
            fft_size or signal_object.n_samples,
            window_fnc=window_fnc,
            fft_impl=fft_impl,
        ), self.frame_buffer
