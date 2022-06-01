import json
import math
import os


def generate_from_c():
    import ctypes

    lib = ctypes.cdll.LoadLibrary(os.path.join('AVRMusicTable', 'shared', 'color.so'))

    class RGBStruct(ctypes.Structure):
        _fields_ = [
            ('r', ctypes.c_char),
            ('g', ctypes.c_char),
            ('b', ctypes.c_char),
        ]
    obj = RGBStruct()

    data = {'r': [], 'g': [], 'b': []}
    for x in range(1020):
        lib.fast_hsi(ctypes.c_float(x), ctypes.c_float(255), ctypes.byref(obj))
        data['r'].append(int(obj.r[0]))
        data['g'].append(int(obj.g[0]))
        data['b'].append(int(obj.b[0]))
    return data


def _normalize(r, g, b, intensity=255):
    t_r = r
    t_g = g
    t_b = b
#    if any(v < 0 for v in (t_r, t_g, t_b)):
    _min = min(t_r,t_g,t_b)

    t_r += -_min
    t_g += -_min
    t_b += -_min
    # intensity * 3 = (r + g + b)
    _int_third = (t_r + t_g + t_b) / 3

    _fac = intensity / _int_third
    return (t_r * _fac)/3, (t_g * _fac)/3, (t_b * _fac)/3


def hsi(hue, intensity, saturation=1):

    def cos_over_cos(a, b):
        return math.degrees(math.cos(math.radians(a))) / math.degrees(math.cos(math.radians(b)))

    def fnc_left(range_l, range_h):
        return intensity + intensity * saturation * (
                1 - cos_over_cos(hue - range_l, range_h - hue)
        )
        
    def fnc_right(range_l, range_h):
        return intensity + intensity * saturation * cos_over_cos(hue - range_l, range_h - hue)
    
    def other():
        return intensity - intensity * saturation
    
    if hue == 0:
        r = intensity + 2 * intensity * saturation
        g = other()
        b = other()
    elif 0 < hue < 120:
        r = fnc_right(0, 60)
        g = fnc_left(0, 60)
        b = other()
    elif hue == 120:
        r = other()
        g = intensity + 2 * intensity * saturation
        b = other()
    elif 120 < hue < 240:
        r = other()
        g = fnc_right(120, 180)
        b = fnc_left(120, 180)
    elif hue == 240:
        r = other()
        g = other()
        b = intensity + 2 * intensity * saturation
    else:
        r = fnc_left(240, 300)
        g = other()
        b = fnc_right(240, 300)
        
    return _normalize(r, g, b, intensity)


def patch_data(data_obj, factor, patch_fnc=None):
    _max_val = max(data_obj['r'] + data_obj['g'] + data_obj['b'])

    _half = _max_val / 2

    for idx in range(len(data_obj['r'])):
        _t = abs(data_obj['r'][idx] - _half)
        data_obj['r'][idx] = patch_fnc(data_obj['r'][idx], _max_val, idx)


def show(steps=1, intensity=255, patch_factor=10, patch_fnc=None):
    from copy import deepcopy
    import matplotlib.pyplot as plt

    with open('fromc.json', 'r') as fp:
        c_data = json.load(fp)

    patched_data = deepcopy(c_data)
    patch_data(patched_data, patch_factor, patch_fnc)

    _len = len(c_data['r'])

    _deg = []
    r_vals = []
    g_vals = []
    b_vals = []
    degrees = range(_len)

    for deg, r, g, b in [(deg, *hsi((360 / _len) * deg, 255, 1)) for deg in degrees]:
        _deg.append((360 / _len) * deg)
        r_vals.append(r)
        g_vals.append(g)
        b_vals.append(b)

    fig, ax = plt.subplots()
    ax.plot(_deg, r_vals, color='red', linewidth=0.4)
    ax.plot(_deg, g_vals, color='green', linewidth=0.4)
    ax.plot(_deg, b_vals, color='blue', linewidth=0.4)

    ax.plot(_deg, c_data['r'], color='red', linewidth=1.2)
    ax.plot(_deg, c_data['g'], color='green', linewidth=1.2)
    ax.plot(_deg, c_data['b'], color='blue', linewidth=1.2)

    ax.plot(_deg, patched_data['r'], color='black', linewidth=.8)
    ax.plot(_deg, patched_data['g'], color='green', linewidth=.8)
    ax.plot(_deg, patched_data['b'], color='blue', linewidth=.8)

    plt.show()

