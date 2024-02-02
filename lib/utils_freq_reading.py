import numpy as np


def f_to_key(f):
    if f > 4100:
        return -1
    if f < 28:
        return -2
    return int(12 * np.log2(f / 440) + 48)


def get_maxima(x, y, reference_yf):
    maxima = []
    sensible_unit = max(y) / 5
    for i in range(1, len(y) - 1):
        if y[i] > sensible_unit and y[i] > y[i-1] + sensible_unit and y[i] > y[i+1] + sensible_unit and y[i] > reference_yf[i]:
            maxima.append(x[i])
    return maxima


def print_frequencies(x, y):
    max_amp = max(y)
    i, = np.where(y == max_amp)
    strongest_freq,  = x[i]
    print("The most significant frequencies:")
    significant_freqs = get_maxima(x, y)
    print(f'The biggest amplitude is {max_amp} for {strongest_freq} Hz')
    print("The most significant frequencies:")
    for i in significant_freqs:
        print(i, end='   ')
    print()


CHANNELS = 2


def get_one_channel(signal):
    one_channel = []
    for index, datum in enumerate(signal):
        if index % CHANNELS == 0:
            one_channel.append(datum)
    return one_channel
