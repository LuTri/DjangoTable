# coding: utf-8
def averages(data, parts=14):
    _data = np.absolute(data)
    vals_per_part = math.ceil(len(_data) / parts)
    t = np.array_split(_data, 14)
    full = np.array(np.resize(t[0], (1, vals_per_part)))
    for sub in t[1:]:
        full = np.concatenate((full, np.resize(sub, (1, vals_per_part))), axis=0)
    _averages = np.average(full, axis=1)
    _mid = vals_per_part / 2
    t_points = np.linspace(_mid, vals_per_part * parts - _mid, num=parts, endpoint=True)
    return (t_points, _averages), {'marker': 'o', 'linewidth': 0}
