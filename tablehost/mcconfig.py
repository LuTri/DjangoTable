import json
import logging
import os

from copy import deepcopy


class FileConfig:
    def __init__(self, filename=None, defaults={}, file_traverse_order=None,
                 bubble_reload_to=None, map_get_to='code'):
        self._logger = logging.getLogger('uart_com')
        self._filename = filename
        self._defaults = defaults
        self._file_traverse_order = file_traverse_order
        self._bubble_reload_to = bubble_reload_to
        self._map_get_to = map_get_to
        self.__data = self._load_data()
        self.__mtime = self._read_m_time()

    def _read_m_time(self):
        if os.path.exists(self._filename):
            return os.stat(self._filename).st_mtime
        return None

    def _load_data(self):
        _data = deepcopy(self._defaults)

        self._logger.warning(f'LOADING DATA {self._filename}')
        if os.path.exists(self._filename):
            with open(self._filename, 'r') as fp:
                conf = json.load(fp)
            actual_conf = conf
            if self._file_traverse_order is not None:
                for entry in reversed(self._file_traverse_order):
                    actual_conf = conf.get(entry)
                    if actual_conf:
                        break

            _data.update(actual_conf)

        self._logger.debug(f'{_data}')

        return _data

    @property
    def data(self):
        _mtime = self._read_m_time()
        outdated = self.__mtime != _mtime
        if outdated:
            self._logger.info(f'Reloading config {self._filename}.')
            self.__data = self._load_data()
            self.__mtime = _mtime

            if self._bubble_reload_to is not None:
                if type(self._bubble_reload_to) in [tuple, list, set]:
                    for item in self._bubble_reload_to:
                        item.reload()
                else:
                    self._bubble_reload_to.reload()
        return self.__data

    def get(self, *args, **kwargs):
        if self._map_get_to is not None:
            return self.data.get(*args, **kwargs).get(self._map_get_to)
        return self.data.get(*args, **kwargs)

    def items(self):
        if self._map_get_to is not None:
            for key in self.data.keys():
                yield key, self.data.get(key).get(self._map_get_to)
        else:
            return super().data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        if self._map_get_to is not None:
            for key in self.data.keys():
                yield self.data.get(key).get(self._map_get_to)
        else:
            return super().data.values()

    def reverse(self, value):
        candidates = []
        for key, _value in self.items():
            try:
                if _value.encode() in value:
                    candidates.append(key)
            except AttributeError:
                pass
        return candidates

    def add_bubble_up(self, item):
        if self._bubble_reload_to is None:
            self._bubble_reload_to = []
        elif type(self._bubble_reload_to) in [set, tuple]:
            self._bubble_reload_to = list(self._bubble_reload_to)
        elif not isinstance(self._bubble_reload_to, list):
            self._bubble_reload_to = [self._bubble_reload_to]
        self._bubble_reload_to.append(item)

    def rm_bubble_up(self, item):
        if self._bubble_reload_to == item:
            self._bubble_reload_to = None
        else:
            self._bubble_reload_to.remove(item)