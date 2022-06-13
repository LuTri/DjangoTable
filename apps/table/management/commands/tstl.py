import logging
import warnings

from functools import wraps

from datetime import datetime
from datetime import timedelta
import time

from django.utils import autoreload
from django.conf import settings
from libs.vbanRelay import VBANCollector

from tablehost.uart import UartGetState
from tablehost.uart import PatchedSerial

from .basePsd import Command as PsdCommand


def fps_sync(fnc):
    if settings.STL_UART_FPS is None:
        return fnc

    delta_t = timedelta(seconds=1 / settings.STL_UART_FPS)

    def wait(callee, logger):
        exec_time = getattr(callee, '_exec_time', None)

        earliest = datetime.now() - delta_t
        if exec_time is not None and exec_time > earliest:
            d_t = (datetime.now() - earliest).total_seconds()
            time.sleep(d_t)
        elif exec_time is not None:
            d_t = (datetime.now() - exec_time).total_seconds()

            notify_cnt = getattr(callee, '_notify_cnt', 0)
            notify_cnt += 1
            setattr(callee, '_notify_cnt', notify_cnt)

            _avg = getattr(callee, '_avg_fps', None)
            if _avg is None:
                _avg = d_t
            else:
                _avg = (_avg + d_t) / 2

            setattr(callee, '_avg_fps', _avg)
            if notify_cnt % 500 == 0:
                getter = UartGetState(serial_class=PatchedSerial)
                _, status = getter.command()
                getter.connection.close()

                _err = getattr(callee, '_errors', 0)
                _err_ratio = _err / notify_cnt * 100


                logger.info(f'Average FPS during last 500 frames: {1 / d_t:.2}, errors: {_err}, rate: {_err_ratio:.4}')

        setattr(callee, '_exec_time', datetime.now())

    @wraps(fnc)
    def wrapper(instance, *args, **kwargs):
        wait(fnc, instance.logger)
        return fnc(instance, *args, **kwargs)
    return wrapper


class Command(VBANCollector, PsdCommand):
    help = "Generate and interactive PSD plotter for streamed PCM data."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.live = True
        self._verbose = False

        self._kwargs.update({'required_samples': self.n_samples_processed})

        self.__live_frame = None
        self.last_frame = 0

        self.logger = logging.getLogger(settings.STL_CMD_LOGGER)
        self.last_cmd_time = None
        self.delta_t = timedelta(seconds=1 / settings.STL_UART_FPS)
        self.cmd_counter = 0
        self.avg_fps = None

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
                self.logger.warning(f'Frames skipped: {skipped}')
            self.last_frame = _last_frame

            self.combined_frames = frames
            if self.needs_setup:
                self.do_setup()

            _start = None
            d_t = 0
            if self.last_cmd_time is not None:
                earliest = self.last_cmd_time + self.delta_t
                _start = datetime.now()
                if earliest > _start:
                    d_t = (earliest - _start).total_seconds()
                    time.sleep(d_t)

            self.present_frame(-1)
            _end = datetime.now()
            if self.last_cmd_time is not None:
                self.avg_fps = self.avg_fps or 1 / (_end - self.last_cmd_time).total_seconds()
                self.avg_fps += 1 / (_end - self.last_cmd_time).total_seconds()
                self.avg_fps /= 2
            self.last_cmd_time = _end

            self.cmd_counter += 1
            if self.cmd_counter % 300 == 0:
                self.logger.info(f'Average FPS during last 300 frames: {self.avg_fps:.2}, seconds waited last frame: {d_t}, Execution time last cmd: {(_end - _start).total_seconds()}')

    def inner_run(self, continuous=False, **options):
        self.logger.info(f'Config:')
        self.logger.info(f'{settings.PRESENTER_VALUE_LIMITS=}')
        self.logger.info(f'{settings.PRESENTER_FREQUENCY_RANGE=}')
        self.logger.info(f'{settings.VBAN_SAMPLES_PROCESSED=}')

        self.run()

        try:
            if continuous:
                while True:
                    self.run_once()
            else:
                self.do_setup()
                self.plotter.run()
        except Exception as exc:
            self.logger.fatal(exc)
        finally:
            self.quit()

    def handle(self, *args, auto_reload=None, **options):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
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
