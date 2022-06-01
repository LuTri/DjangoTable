import math

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal


with open('testPcmStreamsNEW.tps', 'rb') as fp:
    TEST_STREAMS = pickle.load(fp)

t_data = np.array([])
for item in TEST_STREAMS:
    t_data = np.concatenate((t_data, item.mean_channels()))


def overlap_sampling(data, num, sample_rate, ref=0x7FFF, overlap=.61, n_plots=None):
    idx_plot = 0
    n_overlap = math.ceil(overlap * num)

    data_idx = 0
    while data_idx + num + n_overlap < len(data) and (n_plots is None or idx_plot < n_plots):
        freqs, times, sX = signal.spectrogram(
            data[data_idx: data_idx + num + n_overlap],
            window=np.kaiser(num, 4),
            fs=sample_rate,
            nperseg=num,
            noverlap=n_overlap
        )

        avg_s = np.mean(sX, axis=1)
        plt.plot(freqs, 20 * np.log10(avg_s / ref), label=f'Plot Nr {idx_plot}')
        plt.legend()
        plt.show()

        idx_plot += 1
        data_idx += 1


def make_sinoids(duration, sampling_rate, frequencies=[], ref=0x7FFF):
    res = None
    samples = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    for f, amplitude_factor in frequencies:
        signal = np.sin(2 * np.pi * f * samples)
        signal = signal * amplitude_factor
        if res is None:
            res = signal
        else:
            res = res + signal
    return np.int16(res * ref / len(frequencies))


def dbfft(x, fs, win=None, ref=32768):
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

    N = len(x)  # Length of input sequence

    if win is None:
        win = np.ones(N)
    if len(x) != len(win):
            raise ValueError('Signal and window must be of the same length')
    x = x * win

    # Calculate real FFT and frequency vector
    sp = np.fft.rfft(x)
    freq = np.arange((N / 2) + 1) / (float(N) / fs)

    # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum.
    s_mag = np.abs(sp) * 2 / np.sum(win)

    # Convert to dBFS
    s_mag[~np.isfinite(s_mag)] = 1.0
    s_mag[s_mag == 0.] = 1.0

    s_dbfs = 20 * np.log10(s_mag/ref)

    return freq, s_dbfs


def fft_foobar(signal, num, fs):
    for window, label, colors in [(np.hanning(num), 'Hanning', ('#FF0000', '#FF0033')), (np.hamming(num), 'Hamming', ('#00FF00', '#33FF00')), (np.kaiser(num, 4), 'Kaiser', ('#0000FF', '#0033FF'))]:

        _freq, _dbfs = dbfft(signal, fs, window)
        plt.plot(_freq, _dbfs, label=f'{label} window', color=colors[0], linewidth=.4)
    plt.legend()
    plt.show()


def make_signal(sampling_frequency, duration, frequency):
    num = int(sampling_frequency * duration)
    t = np.arange(num, dtype=float) / sampling_frequency

    sensitivity = 0.004  # Sensitivity of the microphone [mV/Pa]
    p_ref = 2e-5 # Reference acoustic pressure [Pa]
    amp = sensitivity * np.sqrt(2)  # Amplitude of sinusoidal signal with RMS of 4 mV (94 dB SPL)
    signal = amp * np.sin(2 * np.pi * frequency * duration * t)  # Signal [V]
    return signal, sampling_frequency, p_ref, num, sensitivity


def whattheactualfick(data, fs, p_ref, num, sensitivity):
    # Calculate the level from time domain signal
    rms_time = np.sqrt(np.mean(data**2))
    db_time = 20 * np.log10(rms_time / sensitivity / p_ref)
    
# Apply window to the signal
    win = np.hamming(num)
    data = data * win

# Get the spectrum and shift it so that DC is in the middle
    spectrum = np.fft.fftshift( np.fft.fft(data) )
    freq = np.fft.fftshift( np.fft.fftfreq(num, 1 / fs) )

# Take only the positive frequencies
    spectrum = spectrum[num//2:]
    freq = freq[num//2:]

# Since we just removed the energy in negative frequencies, account for that
    spectrum *= 2
# If there is even number of samples, do not normalize the Nyquist bin
    if num % 2 == 0:
        spectrum[-1] /= 2

# Scale the magnitude of FFT by window energy
    spectrum_mag = np.abs(spectrum) / np.sum(win)

# To obtain RMS values, divide by sqrt(2)
    spectrum_rms = spectrum_mag / np.sqrt(2)
# Do not scale the DC component
    spectrum_rms[0] *= np.sqrt(2) / 2

# Convert to decibel scale
    spectrum_db = 20 * np.log10(spectrum_rms / sensitivity / p_ref)

# Compare the outputs
    print("Difference in levels: {} dB".format(db_time - spectrum_db.max()))

    plt.plot(freq, spectrum_db)
    plt.xlim((1, fs/2))
    plt.ylim((0, 120))
    plt.grid('on')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB]")
    plt.show()
