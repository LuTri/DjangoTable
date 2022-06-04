import warnings
import struct


class ConversionError(Exception):
    pass


class Converter:
    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self._forward(*args, **kwargs)


class SinglebyteC(Converter):
    MAX_UNSIGNED = (1 << 8) - 1

    def _forward(self, val_8bit):
        result = val_8bit & 0xff
        if (val_8bit ^ self._reverse(result)) > 0:
            raise ConversionError(f'{val_8bit} can not be represented as 1 byte!')
        return result

    def _reverse(self, _byte):
        return _byte

    def reverse(self, _byte):
        return self._reverse(_byte)


class PerConst1Byte(SinglebyteC):
    DIVIDER = NotImplemented

    def _forward(self, real):
        val_8bit = int((self.MAX_UNSIGNED / self.DIVIDER) * real)
        return super()._forward(val_8bit)

    def reverse(self, _byte):
        _byte = super()._reverse(_byte)
        return _byte / float(self.MAX_UNSIGNED) * self.DIVIDER


class Real1Byte(PerConst1Byte):
    DIVIDER = 1.0


class DualbyteC(Converter):
    MAX_UNSIGNED = (1 << 16) - 1

    def _forward(self, val_16bit):
        result = (val_16bit >> 8) & 0xff, val_16bit & 0xff
        if (val_16bit ^ self._reverse(result)) > 0:
            raise ConversionError(f'{val_16bit} can not be represented as 2 bytes!')
        return result

    def _reverse(self, v_bytes):
        _next = 0
        result = 0
        for byte in reversed(v_bytes):
            result |= (byte << _next)
            _next += 8
        return result

    def reverse(self, v_bytes):
        return self._reverse(v_bytes)


class TriplebyteConverter(DualbyteC):
    MAX_UNSIGNED = (1 << 14) - 1

    def _forward(self, val_24bit):
        result = (val_24bit >> 16) & 0xff, (val_24bit >> 8) & 0xff, val_24bit & 0xff
        if (val_24bit ^ self._reverse(result)) > 0:
            raise ConversionError(f'{val_24bit} can not be represented as 2 bytes!')
        return result


class PerConst2Byte(DualbyteC):
    DIVIDER = NotImplemented

    def _forward(self, real):
        val_16bit = int((self.MAX_UNSIGNED / self.DIVIDER) * real)
        return super()._forward(val_16bit)

    def reverse(self, v_bytes):
        _16bit = super()._reverse(v_bytes)
        return _16bit / float(self.MAX_UNSIGNED) * self.DIVIDER


class FullFloat(Converter):
    def _forward(self, value):
        f_bytes = [int(c) for c in struct.pack('f', value)]
        _reversed = self.reverse(f_bytes)
        if _reversed != value:
            warnings.warn(f'Precision lost in conversion! input: {value}, '
                          f'converted: {_reversed}')
        return f_bytes

    def reverse(self, v_bytes):
        return struct.unpack('f', bytes(v_bytes))[0]


class Real360in2Byte(PerConst2Byte):
    DIVIDER = 360.0


class Percent2Byte(PerConst2Byte):
    DIVIDER = 100.0


class Real2Byte(PerConst2Byte):
    DIVIDER = 1.0


full_float = FullFloat()
dualbyte = DualbyteC()
triplebyte = TriplebyteConverter()
per_one_1byte = Real1Byte()
per_one_2byte = Real2Byte()
real_360_2byte = Real360in2Byte()
per_cent_2byte = Percent2Byte()


def stringint_array(value):
    res = []
    _chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789= .,"\'?!'
    for c in value:
        idx = _chars.find(c)
        if idx > -1:
            res.append(idx)

    result_length = len(res)
    if result_length != len(value):
        warnings.warn(f'Not all characters were mapped, skipping '
                      f'{len(value) - result_length} bytes.')
    return [(result_length >> 8) & 0xff, result_length & 0xff] + res
