import numpy as np
import math
import threading
import colorsys
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import warnings

from django.conf import settings
from scipy import signal
from matplotlib.collections import CircleCollection
from matplotlib import widgets
import matplotlib.gridspec as gridspec


mpl.rcParams['savefig.pad_inches'] = 0


SEC_COLORS = list(reversed([
    .0,
    20.0,
    62.0,
    115.0,
    120.0,
    120.0,
    120.0,
    120.0,
]))

AVAILABLE_WINDOWS = [
    'boxcar',
    'triang',
    'blackman',
    'hamming',
    'hann',
    'bartlett',
    'flattop',
    'parzen',
    'bohman',
    'blackmanharris',
    'nuttall',
    'barthann',
    'cosine',
    'exponential',
    'tukey',
    'taylor',

]

def get_section_color(value):
    _step = 0xFFFF / 8
    _refs = np.linspace(-(_step / 2), 0xFFFF - (_step / 2), num=8, endpoint=True)

    col_idx = np.searchsorted(_refs, value) - 1
    hue = SEC_COLORS[col_idx] / 360.0

    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return f'#{hex(int(r * 0xff))[2:]:>02}{hex(int(g * 0xff))[2:]:>02}{hex(int(b * 0xff))[2:]:>02}'


def kaiser_proxy(num):
    return np.kaiser(num, 4)


def raise_90(data):
    return data + 90


def log2_modulator(max_val):
    def modulate(data):
        mod = np.log2(np.linspace(1, max_val, num=129, endpoint=True))
        return data * (mod / np.max(mod))
    return modulate


def NO_WINDOW(m):
    return None


def choose_window(window):
    print(f'{window=}')


class Pyplotter:
    def __init__(self, ion=True, n_frames=None, frame_update_fnc=None, window_update_fnc=None, modulator_update_fnc=None, *args, **kwargs):
        if ion:
            plt.ion()
        self.ion = ion

        self._fig = plt.figure()

        gs0 = gridspec.GridSpec(2, 1, self._fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0])
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])

        self.parent_ax = self._fig.add_subplot(gs00[-1, :-2])
        self.dropdown_ax = self._fig.add_subplot(gs00[-1, -2])
        self.modulator_ax = self._fig.add_subplot(gs00[-1, -1])

        self.frame_ax = self._fig.add_subplot(gs01[0])

        plt.tight_layout()

        #self.widget_ax = plt.axes([0.25, 0.1, 0.65, 0.03],
#                                  facecolor='lightgoldenrodyellow')
        self.widgets = {
            'window': widgets.RadioButtons(self.dropdown_ax, labels=AVAILABLE_WINDOWS, active=0),
            'frame': widgets.Slider(self.frame_ax, 'Frame nr:', valmin=0, valmax=n_frames - 1, valstep=1),
            'modulator': widgets.RadioButtons(self.modulator_ax, labels=['rss', 'max', 'mean', 'abs_mean', 'median', 'abs_median', 'weighted_avg'], active=2)

        }
        self.widgets['frame'].on_changed(frame_update_fnc)
        self.widgets['window'].on_clicked(window_update_fnc)
        self.widgets['modulator'].on_clicked(modulator_update_fnc)

        self.data = []
        self._orig_max = None
        self._orig_min = None
        self._frequencies = None
        self._min = None
        self._max = None

    def do_plot(self, frequency_domain_data, originals):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._min = min([np.min(self.data), np.min(originals)] + ([self._min] if self._min is not None else []))
            self._max = max([np.max(self.data), np.max(originals)] + ([self._max] if self._max is not None else []))
            self.parent_ax.cla()

            _points = np.stack([np.array(self._frequencies).flatten(), np.array(self.data).flatten()])
            segments = np.swapaxes(_points, 1, 0)
            lc = CircleCollection([90], offsets=segments, transOffset=self.parent_ax.transData, facecolors=[get_section_color(v) for v in self.data])
            self.parent_ax.plot(frequency_domain_data.sampled_frequencies, originals * (np.max(self.data) / np.max(originals)), color='red')
            self.parent_ax.add_collection(lc)
            self.parent_ax.plot(self._frequencies, self.data)
            #self._ax.plot(self._frequencies, self.data, 'o', fillstyle='full', color=)
            #
            # _ref = np.max(np.array(self.data))
            #
            # for offset, col in [(0, '#ff5555'), (1000, '#55ff55'), (10000, '#5555ff'), (30000, '#ff00ff'), (50000, '#00ffff')]:
            #     abs_val_range = 0xffff + offset
            #     scaler = abs_val_range / 0xffff
            #
            #     scaled = np.array(self.data) * scaler - offset
            #
            #     scaled_max_diff = np.abs(np.max(scaled) - _ref)
            #
            #     plt.plot(self._frequencies, scaled + scaled_max_diff, color=col, label=f'offset: {offset}')

            self.parent_ax.hlines([0xffff / 8 * x for x in range(8)], 0, max(self._frequencies))

            self.parent_ax.vlines(self._frequencies, 0, 0xFFFF)
            #plt.vlines(frequency_domain_data.sampled_frequencies, 0xffff, 0xffff + 20000, colors=['red'])
            #plt.vlines(self._frequencies + (frequency_domain_data.bar_width / 2), 0, 0xffff + 20000, colors=['#00FF0088'])

            #for idx, vline_value in enumerate(self._frequencies):
            #    plt.annotate(text=f'{idx}: {vline_value:.1f}', xy=(vline_value - 40, 0),
            #               rotation=300)

            #for idx, vline_value in enumerate(frequency_domain_data.sampled_frequencies):
            #    plt.annotate(text=f'{idx}: {vline_value:.1f}', xy=(vline_value - 40, 0xffff - 200),
            #                rotation=300)

            self.parent_ax.set_ylim((0, self._max))
            #plt.legend()
            self.parent_ax.set_xlim(settings.PRESENTER_FREQUENCY_RANGE)
            plt.show()
            if self.ion:
                plt.pause(.03)

    def handle(self, frequency_domain_data):
        self._frequencies = frequency_domain_data.bar_frequencies
        frequency_domain_data.write(self)

        originals = np.array(frequency_domain_data._data)
        _max = np.max(originals)
        _min = np.min(originals)

        if self._orig_max is None or self._orig_max < _max:
            self._orig_max = _max
            print(f'New orig max: {_max}')

        if self._orig_min is None or self._orig_min > _min:
            self._orig_min = _min
            print(f'New orig min: {_min}')

        self.do_plot(frequency_domain_data, originals)

    def command(self, *args, **kwargs):
        self.data = [kwargs[key] for key in sorted([key for key in kwargs.keys() if key.startswith('val_')], key=lambda k: int(k[4:]))]


class PyplotThread(Pyplotter, threading.Thread):
    PLOT_FREQUENCY_LIMITS = (60, 12000)

    WINDOW_FUNCTIONS = [
        #NO_WINDOW,
        #np.hanning,
        #np.hamming,
        #kaiser_proxy,
        #np.blackman,
        signal.windows.flattop,
        #signal.windows.cosine,
    ]

    PLOT_MODULATORS = [
        raise_90,
        log2_modulator(2048),
    ]

    def plotcolor(self, idx):
        f = idx / 8
        rgb = colorsys.hsv_to_rgb(f, 1, 1)

        return (
            f'#{hex(int(rgb[0] * 255))[2:]:>02}'
            f'{hex(int(rgb[1] * 255))[2:]:>02}'
            f'{hex(int(rgb[2] * 255))[2:]:>02}'
        )

    def __init__(self, caller, average_over=4, skip_frames=25,
                 plot_bars=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger('uart_threads')

        self.average_over = average_over
        self.skip_frames = skip_frames

        self.parsed_frames = 0
        self.last_analyzed_frame = 0
        self.last_handled_frame = 0
        self.awaited_frame = skip_frames + average_over

        self.caller = caller
        self.previous_plot_time = None
        self.running = True
        self.last_frame = None

        self.analyzing = True

        self.frame_buffer = np.array([])
        self.frame_buffer2 = np.array([])
        self.frame_buffer_mean = np.array([])
        self.dbfs_buffer = {}
        self._frequencies = None
        self._last_pcm = None
        self._i_pcm_start = 0
        self._i_pcm_step = 1 / average_over

        self.plot_bars = plot_bars

        plt.ion()

        self._figure = None
        self.fourier_plt = None
        self.raw_plt = None
        self.logger.info(f'Fully initialized {self.__class__.__name__}')

        self._norms_idx = 0
        self._norms = ['backward', 'ortho', 'forward']
        self._norm_counter = 0
        self._norm = 'backward'

    @property
    def norm(self):
        self._norm_counter += 1
        if self._norm_counter % 300 == 0:
            self._norms_idx += 1
            self._norms_idx = self._norms_idx % 3
            self._norm = self._norms[self._norms_idx]
            self.logger.info(f'Switched to "{self._norm}" norming.')
        return self._norm

    def dbfft(self, raw_data, fs, win=None, ref=32768, norm=None):
        """
        Calculate spectrum in dB scale
        Args:
            x: input signal
            fs: sampling frequency
            win: vector containing window samples (same length as x).
                 If not provided, then rectangular window is used by default.
            ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

        Returns:
            freq: frequency vector
            s_db: spectrum in dB scale
        """
        num = len(raw_data)  # Length of input sequence

        if win is None:
            win = np.ones(num)
        if len(raw_data) != len(win):
            raise ValueError('Signal and window must be of the same length')
        windowed = raw_data * win

        # Calculate real FFT and frequency vector
        sp = np.fft.rfft(windowed, norm=norm)
        freq = np.arange((num / 2) + 1) / (float(num) / fs)

        # Scale the magnitude of FFT by window and factor of 2,
        # because we are using half of FFT spectrum.
        s_mag = np.abs(sp) * 2 / np.sum(win)

        # Convert to dBFS
        s_mag[~np.isfinite(s_mag)] = 1.0
        s_mag[s_mag == 0.] = 1.0

        s_dbfs = 20 * np.log10(s_mag / ref)

        return freq, s_dbfs

    def wait_for_desired_frame(self):
        frame_available = self.caller.get_frame_counter()
        if frame_available is None or frame_available < self.awaited_frame:
            # self.logger.warning(
            #     f'get_frame_counter returned unsatisfying value: {frame_available} '
            #     f'(waiting for {self.awaited_frame})'
            # )
            return False
        return True

    def process_data(self):
        if self._last_pcm is None or len(self.frame_buffer) == 0:
            return False

        # self.logger.info(f'Attempting to analyze {(len(self.frame_buffer))} bytes of PCM. {self._i_pcm_start=}, {self._last_pcm.n_samples=}')
        target_i = self._i_pcm_start + self._last_pcm.n_samples
        while target_i < len(self.frame_buffer):
            for fnc in self.WINDOW_FUNCTIONS:
                self._frequencies, s_dbfs = self.dbfft(
                    self.frame_buffer[self._i_pcm_start:target_i],
                    self._last_pcm.sample_rate,
                    fnc(self._last_pcm.n_samples),
                    #norm=self.norm,
                )
                # self.logger.info(f'New dbfs item: {s_dbfs.shape}')
                self.dbfs_buffer.setdefault(fnc.__name__, []).append(s_dbfs)
                # self._frequencies, s_dbfs = self.dbfft(
                #     self.frame_buffer_mean[self._i_pcm_start:target_i],
                #     self._last_pcm.sample_rate,
                #     fnc(self._last_pcm.n_samples),
                #     # norm=self.norm,
                # )
                # # self.logger.info(f'New dbfs item: {s_dbfs.shape}')
                # self.dbfs_buffer.setdefault(f'{fnc.__name__}_MEAN', []).append(s_dbfs)
            self._i_pcm_start += math.floor(self._last_pcm.n_samples * self._i_pcm_step)
            target_i = self._i_pcm_start + self._last_pcm.n_samples

            # self.logger.info(
            #     f'Analyzation info:'
            #     f'current dbfs buffers: {len(self.dbfs_buffer)} - '
            #     f'current framebuffer raw: {len(self.frame_buffer)} - '
            #     f'target slice: {target_i} - '
            #     f'expeccted target slice: {self._last_pcm.n_samples * self.average_over} - '
            # )
        if target_i == len(self.frame_buffer) and target_i == self._last_pcm.n_samples * self.average_over:
            return True
        return False

    def get_data(self):
        if self.parsed_frames >= self.average_over:
            return True

        t_frames = self.caller.last_n_pcm(self.average_over)

        #self.logger.warning(f'Fetcheds frames: {len(t_frames)}')

        self.parsed_frames += len(t_frames)
        for item in t_frames[:self.average_over]:
            self.frame_buffer = np.concatenate((
                self.frame_buffer,
                #item.mean_channels(),
                item.data[:, 0],
            ))
            self.frame_buffer2 = np.concatenate((
                self.frame_buffer2,
                #item.mean_channels(),
                item.data[:, 1],
            ))
            self.frame_buffer_mean = np.concatenate((
                self.frame_buffer_mean,
                item.mean_channels(),
            ))
            self._last_pcm = item

        return self.parsed_frames >= self.average_over

    def get_bar_frequencies_adjacent_weights(self, n_bars):
        bar_frequencies, s_size = np.linspace(*self.PLOT_FREQUENCY_LIMITS,
                                              num=n_bars, endpoint=False,
                                              retstep=True)

        frequency_step = self._frequencies[1] - self._frequencies[0]

        _left_nearest = np.searchsorted(self._frequencies, bar_frequencies)
        _right_nearest = np.searchsorted(self._frequencies, bar_frequencies + s_size)
        bar_frequencies += (s_size/2)
        _mid_nearest = np.searchsorted(self._frequencies, bar_frequencies)

        indices = np.array([_left_nearest, _right_nearest])

        refs = np.array([
            np.absolute(self._frequencies[indices[0, :] - 1] - (bar_frequencies - (s_size / 2))) / frequency_step,  # left bar limit distance to next lower frequency
            np.absolute(self._frequencies[indices[1, :] - 1] - (bar_frequencies + (s_size / 2))) / frequency_step,  # right bar limit distance to next lower frequency
        ])

        indices = np.swapaxes(indices, 0, 1)

        abs_distances = [
            np.absolute(np.array(
                self._frequencies[slice(*indices[idx] - 1)]
            ) - bar) for idx, bar in enumerate(bar_frequencies)
        ]

        return bar_frequencies, s_size, indices, refs, abs_distances

    def do_plot(self):
        if self.fourier_plt is None:
            self._figure, (self.fourier_plt, self.raw_plt) = plt.subplots(
                2, 1, gridspec_kw={'height_ratios': [3,1]}, sharey=False
            )

        self.fourier_plt.cla()
        self.raw_plt.cla()

        if self.plot_bars:
            parts = self.caller.parts
            bar_frequencies, bar_width, ref_indices, ref_weights, abs_distances = self.get_bar_frequencies_adjacent_weights(parts)

        for idx, fnc_name in enumerate(self.dbfs_buffer.keys()):
            ref_data = np.array(self.dbfs_buffer[fnc_name])

            if self.caller.apply_modulators:
                for mod in self.PLOT_MODULATORS:
                    ref_data = mod(ref_data)

            if self.plot_bars:
                bar_data = {'avg': [], 'max': [], 'min': [], 'wvg': []}
                _bar_sections = []
                for bar in range(parts):
                    #self.logger.info(f'Slice: {s}')
                    _data = ref_data[:, slice(*ref_indices[bar]-1)]

                    #self.logger.info(f'Data: {_data}')
                    _avg_data = [
                        np.mean(_data[:, 0]),
                        np.mean(_data[:, 1:-1].flatten()),
                        np.mean(_data[:, -1]),
                    ]

                    relevant_data = np.average(
                        np.array(ref_data)[:, slice(*ref_indices[bar] - 1)],
                        axis=0
                    )
                    full_weighted_avg = np.average(relevant_data,
                                                   weights=abs_distances[bar])

                    _weights = [ref_weights[:, bar][0], 1, ref_weights[:, bar][-1]]

                    _bar_sections.append(sorted([np.max(_data[:, 1:-1]), full_weighted_avg, np.min(_data[:, 1:-1])]))

                    bar_data['max'].append(np.max(_data[:, 1:-1]))   # max/min swapped since ^-1
                    #bar_data['avg'].append(np.average(_avg_data, weights=_weights))
                    bar_data['wvg'].append(full_weighted_avg)
                    bar_data['min'].append(np.min(_data[:, 1:-1]))   # max/min swapped since ^-1

                for bar_idx, triplet in enumerate(_bar_sections):
                    bottom = 0
                    _offset = 0
                    # values somewhere between -50 / 65
                    # self.logger.error(f'VALUES TO BAR: {list(triplet)}')
                    for tr_idx, val in enumerate(triplet):
                        bar = self.fourier_plt.bar(
                            bar_frequencies[bar_idx],
                            val - bottom,
                            bar_width,
                            color=self.plotcolor(tr_idx),
                            edgecolor='black',
                            bottom=bottom + _offset,
                        )
                        self.fourier_plt.bar_label(bar, padding=-10, color=self.plotcolor(6 - tr_idx))
                        bottom += (val - bottom)
                # self.fourier_plt.bar(bar_frequencies, np.array(bar_data['max']),
                #                      bar_width, color=self.plotcolor(idx + .2),
                #                      edgecolor='black', bottom=-90, alpha=.5)
                # self.fourier_plt.bar(bar_frequencies, np.array(bar_data['wvg']),
                #                      bar_width, color='blue',  # self.plotcolor(idx),
                #                      edgecolor='black', bottom=-90)
                # self.fourier_plt.bar(bar_frequencies, np.array(bar_data['avg']),
                #                      bar_width, color='yellow',  # self.plotcolor(idx),
                #                      edgecolor='black', bottom=-90, alpha=.5)
                # self.fourier_plt.bar(bar_frequencies, np.array(bar_data['min']),
                #                      bar_width, color=self.plotcolor(idx + .4),
                #                      edgecolor='black', bottom=-90, alpha=.5)

            self.fourier_plt.plot(
                self._frequencies,
                np.mean(ref_data, axis=0),
                label=f'W-fnc: {fnc_name}',
                color=self.plotcolor(idx)
             )

        self.fourier_plt.legend(frameon=False, loc='lower center',
                                ncol=len(self.WINDOW_FUNCTIONS) * 2)

        self.fourier_plt.set_ylim(-120, 80)
        self.fourier_plt.set_xlim(*self.PLOT_FREQUENCY_LIMITS)

        self.raw_plt.plot(range(len(self.frame_buffer)), self.frame_buffer, color='green')
        self.raw_plt.plot(range(len(self.frame_buffer2)), self.frame_buffer2, color='blue')
        self.raw_plt.plot(range(len(self.frame_buffer_mean)), self.frame_buffer_mean, color='red')

        self.raw_plt.set_ylim(-32767, 32767)

        #plt.subplots_adjust(wspace=0, hspace=0)
        #plt.autoscale(tight=True)
        plt.show()  # shows the plot

        plt.pause(1 / self._last_pcm.sample_rate)

        self.frame_buffer = np.array([])
        self.frame_buffer2 = np.array([])
        self.frame_buffer_mean = np.array([])
        self.dbfs_buffer = {}
        self._i_pcm_start = 0

        self.parsed_frames = 0
        #self.last_analyzed_frame = 0
        #self.last_handled_frame = 0

        self.last_handled_frame = self._last_pcm.frame_counter

        self.awaited_frame = self.last_handled_frame + self.skip_frames + self.average_over

    def run(self):
        while self.running:
            self.get_data()
            if not self.process_data():
                time.sleep(.02)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.do_plot()

    def quit(self):
        self.running = False
        self.join()
