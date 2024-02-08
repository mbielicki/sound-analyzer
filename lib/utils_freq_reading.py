import numpy as np


def f_to_key(f):
    if f > 4100:
        return -1
    if f < 28:
        return -2
    return int(12 * np.log2(f / 440) + 48)


def get_maxima(x, y):
    maxima = []
    group_count = 5
    sensible_unit = max(max(y)/5, 0.05)
    for i in range(0, len(y) - group_count):
        group = y[i: i + group_count]
        gmax = max(group)
        gmin = min(group)
        diff = gmax - gmin
        if diff > sensible_unit:
            j = np.where(group == gmax)[0]
            maxima.append(x[i+j])

    for i, d in enumerate(maxima):
        if i == 0:
            prev = d
            continue
        if d == prev:
            del maxima[i]
        prev = d

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
