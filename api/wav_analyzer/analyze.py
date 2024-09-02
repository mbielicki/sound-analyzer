import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import rfft, rfftfreq

import os
import time
import threading

type npFloats = npt.NDArray[np.floating]
type npTuple = tuple[npFloats, npFloats]

matplotlib.use('Agg')

SAMPLE_RATE = 48_000 

def extract_frequencies(chunk: bytes) -> npTuple:
        data_np = np.frombuffer(chunk, dtype=np.int16)

        samples_no = len(data_np)
        samples_ps = SAMPLE_RATE

        xf = rfftfreq(samples_no, 1 / samples_ps)
        yf = np.abs(rfft(data_np))
        
        max_freq = 8_000 # higher than B8
        i_max: int = np.where(xf < max_freq)[0][-1]

        xf = xf[:i_max]
        yf = yf[:i_max]

        return xf, yf

def extract_notes(xf: npFloats, yf: npFloats) -> list[int]:        
        piano_keys: list[int] = list()
        for f in get_maxima(xf, yf):
            piano_keys.append(f_to_key(f))

        return piano_keys



def f_to_key(f: float) -> int:
    if f > 4100:
        return -1
    if f < 28:
        return -2
    return int(12 * np.log2(f / 440) + 48)



def get_maxima(x: npFloats, y: npFloats):
    maxima = []
    group_size = 5
    sensible_unit = max(max(y)/20, 0.05e7)
    threshold = 0.1e7
    for i in range(0, len(y) - group_size):
        group = y[i: i + group_size]
        gmax = max(group)
        if gmax < threshold:
             continue
        gmin = min(group)
        diff = gmax - gmin
        if diff < sensible_unit:
            continue
        j, = np.where(group == gmax)[0]
        if j == 0 or j == group_size - 1:
            continue
        maxima.append(x[i+j])

    return list(dict.fromkeys(maxima))

def plot_frequencies(xf: npFloats, yf: npFloats, plot_file: str) -> None:
        fig, ax = plt.subplots()
        ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude')
        ax.set_xscale('log')
        ax.set_xlim(0, 8000)
        ax.set_ylim(0, 1e7)
        ax.plot(xf, yf)
        fig.savefig(plot_file)

        def delete_plot_file():
            time.sleep(5)
            os.remove(plot_file)

        threading.Thread(target=delete_plot_file).start()

        
def remove_wav_headers(chunk: bytes) -> None:
    if chunk[:4] == b'RIFF':
        chunk = chunk[44:]