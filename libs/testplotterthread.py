import numpy as np
import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from django.conf import settings
from scipy import signal

from matplotlib.collections import CircleCollection
from matplotlib import widgets
import matplotlib.gridspec as gridspec

from matplotlib import animation

from libs.spectral import DATA_COMBINER
from libs.spectral import make_linear_output_scaler

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
    def _create_interface_axes(self):
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

        if not self.live:
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

    def _create_interface(self, window_update_fnc, fft_impl_updater,
                          samples_update_fnc, max_range_updater,
                          min_range_updater, n_frames, frame_update_fnc):
        self._create_interface_axes()

        combiners = list(DATA_COMBINER.keys())
        self.widgets = {
            'window': widgets.RadioButtons(self.window_fnc_ax, labels=AVAILABLE_WINDOWS, active=AVAILABLE_WINDOWS.index(settings.VBAN_WELCH_WINDOW)),
            'max_freq': widgets.Slider(self.max_freq_ax, '', valmin=0, valmax=22000, valstep=5,
                                       valinit=self.max_freq),
            'max_freq_tb': widgets.TextBox(self.max_freq_tb_ax, 'Max Freq:', initial=f'{self.max_freq}'),

            'min_freq': widgets.Slider(self.min_freq_ax, '', valmin=0, valmax=22000, valstep=5,
                                       valinit=self.min_freq),
            'min_freq_tb': widgets.TextBox(self.min_freq_tb_ax, 'Min Freq:', initial=f'{self.min_freq}'),
            'new_max_tb': widgets.TextBox(self.new_max_ax, 'n Max:', initial=f'{self.o_scale_new_max}'),
            'new_min_tb': widgets.TextBox(self.new_min_ax, 'n Min:', initial=f'{self.o_scale_new_min}'),
            'old_max_tb': widgets.TextBox(self.old_max_ax, 'o Max:', initial=f'{self.o_scale_old_max}'),
            'old_min_tb': widgets.TextBox(self.old_min_ax, 'o Min:', initial=f'{self.o_scale_old_min}'),
            'modulator': widgets.RadioButtons(self.modulator_ax, labels=combiners, active=combiners.index(settings.RSS_CALCULATOR)),
            'fft_impl': widgets.RadioButtons(self.fft_ax,
                                             labels=['scipy.periodigram', 'kramer', 'scipy.welch', 'foo.fft'],
                                             active=0),
            'n_samples': widgets.TextBox(self.samples_ax, label='Processed samples:',
                                         initial=f'{settings.VBAN_SAMPLES_PROCESSED}'),
        }

        if not self.live:
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

    def __init__(self, live=True, parent_obj=None, n_frames=0,
                 min_freq=None, max_freq=None, frame_update_fnc=None,
                 window_update_fnc=None, modulator_cb=None,
                 samples_update_fnc=None, min_range_updater=None,
                 max_range_updater=None, fft_impl_updater=None,
                 os_new_min=None, os_new_max=None, os_old_min=None,
                 os_old_max=None, spectral_creator=None, *args, **kwargs):

        self.spectral_creator = spectral_creator

        self.parent_obj = parent_obj
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.o_scale_new_min = os_new_min
        self.o_scale_new_max = os_new_max
        self.o_scale_old_min = os_old_min
        self.o_scale_old_max = os_old_max

        self.window_fnc = settings.VBAN_WELCH_WINDOW
        self.fft_impl = 'scipy.periodigram'

        self.raw_plot = None
        self.fft_plot = None

        self.sample_freq_plot = None
        self.bar_border_lines = None
        self.bar_freq_lines = None
        self.unscaled_fft_plot = None
        self.peak_dots = None

        self.audio_data = None
        self.data = []
        self._orig_max = None
        self._orig_min = None
        self._frequencies = None

        self._min = settings.STL_FFT_SCALER_NEW_MIN
        self._max = settings.STL_FFT_SCALER_NEW_MAX

        self._modulator_cb = modulator_cb
        self._modulator = settings.RSS_CALCULATOR

        self.last_frame = 0
        self.presenter = None

        self.live = live

        self._fig = plt.figure()
        self._fig.tight_layout()

        self.last_change = None

        if live:
            self.animator = animation.FuncAnimation(
                self._fig,
                self.get_plot_updater(),
                interval=5,
                blit=True,
                repeat=True,
            )

        self._create_interface(self.get_window_updater(), self.get_impl_updater(),
                               self.get_samples_updater(), self.get_range_updater(False),
                               self.get_range_updater(True), n_frames, frame_update_fnc)

    def run(self):
        self.prepare_plot()
        plt.show()

    def process_frame(self, frame):
        self.handle(*self.spectral_creator(
            frame,
            mean_channels=True,
            window_fnc=self.window_fnc,
            presented_range=[self.min_freq, self.max_freq],
        ))
        self.show_data()

    def update(self, *args, **kwargs):
        try:
            last_frame = self.parent_obj.last_n_pcm(1)[-1]

            if last_frame.frame_counter != self.last_frame:
                try:
                    self.last_frame = last_frame.frame_counter
                    self.process_frame(last_frame)
                except Exception as exc:
                    print(f'Exception was: {exc}')
                    if self.last_change is not None:
                        _name, _val = self.last_change
                        print(f'ERROR: resetting last_change: {_name} -> {_val}')
                        setattr(self, _name, _val)
                    raise
        finally:
            return (
                self.raw_plot,
                self.fft_plot,
                self.unscaled_fft_plot,
                self.sample_freq_plot,
                self.peak_dots,
                self.bar_border_lines,
                self.bar_freq_lines,
            )

    def get_plot_updater(self):
        def proxy(*args, **kwargs):
            return self.update(*args, **kwargs)
        return proxy

    def get_impl_updater(self):
        def updater(fft_impl):
            self.last_change = ('fft_impl', self.fft_impl)
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
            self.last_change = ('window_fnc', self.window_fnc)
            self.window_fnc = fnc_name
        return updater

    def get_samples_updater(self):
        def updater(value):
            self.parent_obj.n_samples_processed = value
            self.parent_obj.quit()
            self.parent_obj._kwargs.update({'required_samples': self.parent_obj.n_samples_processed})
            self.parent_obj.run()

        return updater

    def get_range_updater(self, for_min=False):
        def updater(value):
            if for_min:
                self.last_change = ('min_freq', self.min_freq)
                self.min_freq = value
            else:
                self.last_change = ('max_freq', self.max_freq)
                self.max_freq = value
        return updater

    def get_scaler_updater(self, old=False, for_min=False):
        def updater(value):
            try:
                int_val = int(value)
                if old and for_min:
                    self.last_change = ('o_scale_old_min', self.o_scale_old_min)
                    self.o_scale_old_min = int_val
                elif old:
                    self.last_change = ('o_scale_old_max', self.o_scale_old_max)
                    self.o_scale_old_max = int_val
                elif for_min:
                    self.last_change = ('o_scale_new_min', self.o_scale_new_min)
                    self.o_scale_new_min = int_val
                else:
                    self.last_change = ('o_scale_new_max', self.o_scale_new_max)
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
            self.last_change = ('_modulator', self._modulator)
            self._modulator = modulator
            self._modulator_cb()
        return updater

    def prepare_plot(self):
        self._min = settings.STL_FFT_SCALER_NEW_MIN
        self._max = settings.STL_FFT_SCALER_NEW_MAX

        #_min_r = np.min(frequency_domain_data.bar_borders) - 20
        #_max_r = np.max(frequency_domain_data.bar_borders) + 20

        #self.parent_ax.cla()
        #self.raw_ax.cla()

        #dt = 1/frequency_domain_data.sample_rate
        #dt_x = np.linspace(-dt * len(raw_data), 0, num=len(raw_data))

        self.raw_plot, = self.raw_ax.plot([], [])
        self.raw_ax.set_ylim((-32768, 32768))

        #self.raw_ax.hlines([0], np.min(dt_x), np.max(dt_x), color='red')
        #self.raw_ax.vlines(dt_x[::256], -32768, 32768, color='red')

#       segments = np.swapaxes(np.stack([np.array(self._frequencies).flatten(), np.array(t_plot).flatten()]), 1, 0)
        self.peak_dots = CircleCollection(
            [90],
            offsets=[[0, 0]],
            transOffset=self.parent_ax.transData,
            facecolors=[],
        )

        self.unscaled_fft_plot, = self.parent_ax.semilogx(
            [],  #frequency_domain_data.sampled_frequencies,
            [],  #t_orig,
            color='red',
        )

        self.fft_plot, = self.parent_ax.semilogx(
            [],  # self._frequencies
            [],  # t_plot
        )

        #self.parent_ax.hlines([0xffff / 8 * x for x in range(8)], 0, max(self._frequencies))

        self.sample_freq_plot = self.parent_ax.vlines(
            [],  # frequency_domain_data.sampled_frequencies,
            self._min,
            self._max,
            color='#00000011'
        )
        self.bar_border_lines = self.parent_ax.vlines(
            [],  # frequency_domain_data.bar_borders,
            self._min,
            self._max,
            color='#ff000033'
        )

        self.bar_freq_lines = self.parent_ax.vlines(
            [],  # self._frequencies,
            self._min,
            self._max
        )

        self.parent_ax.set_ylim((self._min, self._max))
        self.parent_ax.set_xlim((self.min_freq, self.max_freq))
        self.parent_ax.add_collection(self.peak_dots)

    def do_plot(self, frequency_domain_data, originals, raw_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            t_plot = np.copy(self.data)
            t_orig = np.copy(originals)

            _min_r = np.min(frequency_domain_data.bar_borders) - 20
            _max_r = np.max(frequency_domain_data.bar_borders) + 20

            self.parent_ax.cla()
            self.raw_ax.cla()

            dt = 1/frequency_domain_data.sample_rate
            dt_x = np.linspace(-dt * len(raw_data), 0, num=len(raw_data))
            self.raw_plot, = self.raw_ax.plot(dt_x, raw_data)

            self.raw_ax.hlines([0], np.min(dt_x), np.max(dt_x), color='red')
            self.raw_ax.vlines(dt_x[::256], -32768, 32768, color='red')

            self.raw_ax.set_ylim((-32768, 32768))

            _points = np.stack([np.array(self._frequencies).flatten(), np.array(t_plot).flatten()])
            segments = np.swapaxes(_points, 1, 0)
            self.peak_dots = CircleCollection([90], offsets=segments, transOffset=self.parent_ax.transData, facecolors=[get_section_color(v) for v in t_plot])

            self.unscaled_fft_plot, self.parent_ax.semilogx(frequency_domain_data.sampled_frequencies, t_orig, color='red')
            self.parent_ax.add_collection(self.peak_dots)

            self.fft_plot, = self.parent_ax.semilogx(self._frequencies, t_plot)

            self.parent_ax.hlines([0xffff / 8 * x for x in range(8)], 0, max(self._frequencies))

            self.sample_freq_plot = self.parent_ax.vlines(frequency_domain_data.sampled_frequencies, self._min, self._max, color='#00000011')
            self.bar_border_lines = self.parent_ax.vlines(frequency_domain_data.bar_borders, self._min, self._max, color='#ff000033')
            self.bar_freq_lines = self.parent_ax.vlines(self._frequencies, self._min, self._max)

            self.parent_ax.set_ylim((self._min, self._max))
            self.parent_ax.set_xlim((_min_r, _max_r))

            self.parent_ax.grid()
            plt.show()
            if self.live:
                plt.pause(dt * 256)

    def _line_data(self, x, v_min, v_max):
        tmp = np.swapaxes(np.stack([
            np.repeat(x, 2).flatten(),
            np.repeat([[v_min, v_max]], [len(x)], axis=0).flatten(),
        ]), 1, 0)

        return np.reshape(tmp, (-1, 2, 2))

    def show_data(self):
        dt = 1 / self.presenter.sample_rate
        dt_x = np.linspace(-dt * len(self.audio_data), 0, num=len(self.audio_data))

        scaler = make_linear_output_scaler(
            self.o_scale_new_max,
            self.o_scale_new_min,
            self.o_scale_old_max,
            self.o_scale_old_min,
            False
        )

        self.raw_plot.set_data(dt_x, self.audio_data)
        self.fft_plot.set_data(self._frequencies, self.data)
        self.unscaled_fft_plot.set_data(self.presenter.sampled_frequencies,
                                        scaler(self.presenter.data))

        self.peak_dots.set_facecolors([get_section_color(v) for v in self.data])
        self.peak_dots.set_offsets(np.swapaxes(np.stack(
            [np.array(self._frequencies).flatten(),
             np.array(self.data).flatten()]
        ), 1, 0))

        self.sample_freq_plot.set_segments(
            self._line_data(self.presenter.sampled_frequencies, self._min,
                            self._max))
        self.bar_border_lines.set_segments(
            self._line_data(self.presenter.bar_borders, self._min, self._max))
        self.bar_freq_lines.set_segments(
            self._line_data(self.presenter.bar_frequencies, self._min,
                            self._max))

        self.parent_ax.set_xlim((self.min_freq, self.max_freq))
        self.raw_ax.set_xlim((np.min(dt_x), np.max(dt_x)))

    def handle(self, frequency_domain_data, raw_data):
        self._frequencies = frequency_domain_data.bar_frequencies
        frequency_domain_data.write(self, self._modulator,
                                    self.o_scale_new_min, self.o_scale_new_max,
                                    self.o_scale_old_min, self.o_scale_old_max)

        _max = np.max(frequency_domain_data.data)
        _min = np.min(frequency_domain_data.data)

        if self._orig_max is None or self._orig_max < _max:
            self._orig_max = _max

        if self._orig_min is None or self._orig_min > _min:
            self._orig_min = _min

        self.presenter = frequency_domain_data
        self.audio_data = raw_data

    def command(self, *args, **kwargs):
        self.data = [kwargs[key]
                     for key in sorted(
                        [key for key in kwargs.keys()
                         if key.startswith('val_')],
                     key=lambda k: int(k[4:]))]
