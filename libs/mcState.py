import re
import ctypes

CONSTANTS = {
    'N_COLS': 14,
    'N_ROWS': 8
}

class StateStruct:
    FINDER = re.compile(r'^\s*(\S*?)(_t)*\s(\S*?)(\[(\S*)\])*;$', re.MULTILINE)
    END_STR = '} STATE;'
    START_STR = 'typedef struct {'

    def __init__(self):
        with open('AVRMusicTable/state.h') as fp:
            self._header = fp.read()
        _end = self._header.find('} STATE;')
        _start = self._header.rfind('typedef struct {', 0, _end)

        self._typedef = self._header[_start + len(self.START_STR):_end]

        self._members = [
                {
                    'type': getattr(ctypes, 'c_' + e[0]),
                    'name': e[2],
                    'quantifier': e[4] or None,
                } for e in self.FINDER.findall(self._typedef)
        ]

    def read(self, data, mapping):
        _loaded = data[0]
        yield 'load_status', _loaded

        data = data[1:]

        for member in self._members:
            _size = ctypes.sizeof(member['type'])

            reverse = mapping.get(member['name'], mapping.get('_default'))

            if member['quantifier'] is None:
                m_data = data[:_size]
                data = data[_size:]
                member['data'] = reverse(m_data)
                member['original'] = m_data

                yield member['name'], member['data']
            else:
                cnt = CONSTANTS[member['quantifier']]
                member['data'] = []
                member['original'] = []
                for idx in range(cnt):
                    member['data'].append(reverse(data[:_size]))
                    member['original'].append(data[:_size])
                    data = data[_size:]
                    yield f'{member["name"]}_{idx}', member['data'][-1]
