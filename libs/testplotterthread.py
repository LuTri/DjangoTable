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
mpl.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['figure.autolayout'] = True

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
    def __init__(self, live=True, n_frames=None, min_freq=None, max_freq=None,
                 frame_update_fnc=None, window_update_fnc=None,
                 modulator_cb=None, samples_update_fnc=None,
                 min_range_updater=None, max_range_updater=None,
                 fft_impl_updater=None, os_new_min=None, os_new_max=None,
                 os_old_min=None, os_old_max=None, *args, **kwargs):

        self.o_scale_new_min = os_new_min
        self.o_scale_new_max = os_new_max
        self.o_scale_old_min = os_old_min
        self.o_scale_old_max = os_old_max
        self.live = live
        if live:
            plt.ion()
        self._fig = plt.figure()

        global_gs = gridspec.GridSpec(10, 1, self._fig)

        self.raw_ax = self._fig.add_subplot(global_gs[:3, -1])

        body_gs = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=global_gs[3:-1, -1])

        wind_scaler_gs = gridspec.GridSpecFromSubplotSpec(15, 1, subplot_spec=body_gs[-1, -2])

        footer_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=global_gs[-1, -1])

        mod_samples_fft_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=body_gs[-1, -1])

        self.parent_ax = self._fig.add_subplot(body_gs[-1, :-2])
        self.window_fnc_ax = self._fig.add_subplot(wind_scaler_gs[:-4, -1])

        self.new_max_ax = self._fig.add_subplot(wind_scaler_gs[-4, -1])
        self.new_min_ax = self._fig.add_subplot(wind_scaler_gs[-3, -1])
        self.old_max_ax = self._fig.add_subplot(wind_scaler_gs[-2, -1])
        self.old_min_ax = self._fig.add_subplot(wind_scaler_gs[-1, -1])

        self.modulator_ax = self._fig.add_subplot(mod_samples_fft_gs[:-4, -1])
        self.fft_ax = self._fig.add_subplot(mod_samples_fft_gs[-4:-1, -1])
        self.samples_ax = self._fig.add_subplot(mod_samples_fft_gs[-1, -1])

        footer_idx = 0

        if not live:
            frame_gs = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=footer_gs[footer_idx, -1])
            footer_idx += 1
            self.frame_ax = self._fig.add_subplot(frame_gs[-1, 1:])
            self.frame_tb_ax = self._fig.add_subplot(frame_gs[-1, 0])

        max_f_gs = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=footer_gs[footer_idx, -1])
        footer_idx += 1
        self.max_freq_ax = self._fig.add_subplot(max_f_gs[-1, 1:])
        self.max_freq_tb_ax = self._fig.add_subplot(max_f_gs[-1, 0])

        min_f_gs = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=footer_gs[footer_idx, -1])
        footer_idx += 1
        self.min_freq_ax = self._fig.add_subplot(min_f_gs[-1, 1:])
        self.min_freq_tb_ax = self._fig.add_subplot(min_f_gs[-1, 0])

        self._fig.tight_layout()

        self.widgets = {
            'window': widgets.RadioButtons(self.window_fnc_ax, labels=AVAILABLE_WINDOWS, active=4),
            'max_freq': widgets.Slider(self.max_freq_ax, '', valmin=0, valmax=22000, valstep=5,
                                       valinit=max_freq),
            'max_freq_tb': widgets.TextBox(self.max_freq_tb_ax, 'Max Freq:', initial=f'{max_freq}'),

            'min_freq': widgets.Slider(self.min_freq_ax, '', valmin=0, valmax=22000, valstep=5,
                                       valinit=min_freq),
            'min_freq_tb': widgets.TextBox(self.min_freq_tb_ax, 'Min Freq:', initial=f'{min_freq}'),
            'new_max_tb': widgets.TextBox(self.new_max_ax, 'n Max:', initial=f'{self.o_scale_new_max}'),
            'new_min_tb': widgets.TextBox(self.new_min_ax, 'n Min:', initial=f'{self.o_scale_new_min}'),
            'old_max_tb': widgets.TextBox(self.old_max_ax, 'o Max:', initial=f'{self.o_scale_old_max}'),
            'old_min_tb': widgets.TextBox(self.old_min_ax, 'o Min:', initial=f'{self.o_scale_old_min}'),
            'modulator': widgets.RadioButtons(
                self.modulator_ax,
                labels=['rss', 'max', 'mean', 'abs_mean', 'median', 'abs_median', 'heavy_varianz', 'ultra_heavy_varianz', 'light_varianz', 'mean_rss'],
                active=6),
            'fft_impl': widgets.RadioButtons(self.fft_ax, labels=['scipy.periodigram', 'kramer', 'scipy.welch', 'foo.fft'], active=0),
            'n_samples': widgets.TextBox(self.samples_ax, label='Processed samples:',
                                         initial=f'{settings.VBAN_SAMPLES_PROCESSED}'),
        }

        if not live:
            self.widgets['frame'] = widgets.Slider(self.frame_ax, '', valmin=0, valmax=n_frames - 1, valstep=1)
            self.widgets['frame_tb'] = widgets.TextBox(self.frame_tb_ax, 'Frame:', initial=f'{0}')
            self.widgets['frame'].on_changed(self.get_related_updater('frame_tb', frame_update_fnc))
            self.widgets['frame_tb'].on_submit(self.get_related_updater('frame', frame_update_fnc))

        self.widgets['new_min_tb'].on_submit(self.get_scaler_updater(False, True))
        self.widgets['new_max_tb'].on_submit(self.get_scaler_updater(False, False))
        self.widgets['old_min_tb'].on_submit(self.get_scaler_updater(True, True))
        self.widgets['old_max_tb'].on_submit(self.get_scaler_updater(True, False))

        self.widgets['window'].on_clicked(window_update_fnc)
        self.widgets['fft_impl'].on_clicked(fft_impl_updater)
        self.widgets['modulator'].on_clicked(self.get_modulator_updater())
        self.widgets['n_samples'].on_submit(self.get_n_samples_updater(samples_update_fnc))

        self.widgets['max_freq'].on_changed(self.get_related_updater('max_freq_tb', max_range_updater))
        self.widgets['max_freq'].slidermin = self.widgets['min_freq']
        self.widgets['max_freq_tb'].on_submit(self.get_related_updater('max_freq', max_range_updater))

        self.widgets['min_freq'].on_changed(self.get_related_updater('min_freq_tb', min_range_updater))
        self.widgets['min_freq'].slidermax = self.widgets['max_freq']
        self.widgets['min_freq_tb'].on_submit(self.get_related_updater('min_freq', min_range_updater))

        self.data = []
        self._orig_max = None
        self._orig_min = None
        self._frequencies = None
        self._min = None
        self._max = None

        self._modulator_cb = modulator_cb
        self._modulator = settings.RSS_CALCULATOR

    def get_scaler_updater(self, old=False, for_min=False):
        def updater(value):
            try:
                int_val = int(value)
                if old and for_min:
                    self.o_scale_old_min = int_val
                elif old:
                    self.o_scale_old_max = int_val
                elif for_min:
                    self.o_scale_new_min = int_val
                else:
                    self.o_scale_new_max = int_val
            except ValueError:
                print(f'Illegal value "{value}"')
        return updater

    def get_related_updater(self, widget_name, par_fnc):
        def updater(value):
            try:
                int_val = int(value)
                _widget = self.widgets[widget_name]

                _widget.set_val(int_val)

                if getattr(par_fnc, '_HANDLED', None) != int_val:
                    setattr(par_fnc, '_HANDLED', int_val)
                    par_fnc(int_val)
            except ValueError:
                print(f'Illegal value "{value}"')
        return updater

    def get_n_samples_updater(self, par_fnc):
        def updater(value):
            try:
                int_val = int(value)
                new_max = par_fnc(int_val)
                if not self.live:
                    slider_widget = self.widgets['frame']
                    slider_widget.set_val(0)
                    slider_widget.valmax = new_max - 1
                    slider_widget.ax.set_xlim(slider_widget.valmin, slider_widget.valmax)
            except ValueError as exc:
                print(exc)
                print(f'Illegal value "{value}"')
        return updater

    def get_modulator_updater(self):
        def updater(modulator):
            self._modulator = modulator
            self._modulator_cb()
        return updater

    def do_plot(self, frequency_domain_data, originals, raw_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            #self._min = min([np.min(self.data), np.min(originals)] + ([self._min] if self._min is not None and self.live else []))
            #self._max = max([np.max(self.data), np.max(originals)] + ([self._max] if self._max is not None and self.live else []))

            #t_plot[~np.isfinite(t_plot)] = -100
            #t_orig[~np.isfinite(t_orig)] = -100

            #_min = min([np.min(t_plot), np.min(t_orig), -1])
            #_max = max([np.max(t_plot), np.max(t_orig), 1])

            self._min = -10
            self._max = 0xFFFF + 10

            t_plot = np.copy(self.data)
            t_orig = np.copy(originals)
            #if np.mean(t_plot) < 0:
            #    t_plot = t_plot + 75
            #    t_orig = t_orig + 75

            _min_r = np.min(self._frequencies) - 20
            _max_r = np.max(self._frequencies) + (frequency_domain_data.bar_width) / 2 + 20

            #while self._min > _min:
            #    self._min -= 200
            #while self._max < _max:
            #    self._max += 200

            self.parent_ax.cla()
            self.raw_ax.cla()

            dt = 1/frequency_domain_data.sample_rate
            dt_x = np.linspace(-dt * len(raw_data), 0, num=len(raw_data))
            self.raw_ax.plot(dt_x, raw_data)

            self.raw_ax.hlines([0], np.min(dt_x), np.max(dt_x), color='red')
            self.raw_ax.vlines(dt_x[::256], -32768, 32768, color='red')

            self.raw_ax.set_ylim((-32768, 32768))

            _points = np.stack([np.array(self._frequencies).flatten(), np.array(t_plot).flatten()])
            segments = np.swapaxes(_points, 1, 0)
            lc = CircleCollection([90], offsets=segments, transOffset=self.parent_ax.transData, facecolors=[get_section_color(v) for v in t_plot])
            self.parent_ax.plot(frequency_domain_data.sampled_frequencies, t_orig, color='red')
            self.parent_ax.add_collection(lc)
            self.parent_ax.plot(self._frequencies, t_plot)

            self.parent_ax.hlines([0xffff / 8 * x for x in range(8)], 0, max(self._frequencies))

            self.parent_ax.vlines(frequency_domain_data.sampled_frequencies, self._min, self._max, color='#00000011')

            self.parent_ax.vlines([np.max(self._frequencies) + frequency_domain_data.bar_width / 2] + [x - (frequency_domain_data.bar_width / 2) for x in self._frequencies], self._min, self._max, color='#ff000033')

            self.parent_ax.vlines(self._frequencies, self._min, self._max)

            self.parent_ax.set_ylim((self._min, self._max))
            self.parent_ax.set_xlim((_min_r, _max_r))
            self.parent_ax.grid()
            plt.show()
            if self.live:
                plt.pause(dt * 256)

    def handle(self, frequency_domain_data, raw_data):
        self._frequencies = frequency_domain_data.bar_frequencies
        frequency_domain_data.write(self, self._modulator,
                                    self.o_scale_new_min, self.o_scale_new_max,
                                    self.o_scale_old_min, self.o_scale_old_max)

        originals = np.array(frequency_domain_data.data)
        _max = np.max(originals)
        _min = np.min(originals)

        if self._orig_max is None or self._orig_max < _max:
            self._orig_max = _max
            print(f'New orig max: {_max}')

        if self._orig_min is None or self._orig_min > _min:
            self._orig_min = _min
            print(f'New orig min: {_min}')

        self.do_plot(frequency_domain_data, originals, raw_data)

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
