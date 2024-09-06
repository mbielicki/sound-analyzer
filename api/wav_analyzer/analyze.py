import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import rfft, rfftfreq
import os
import time
import threading

from api.wav_analyzer.notes import f_to_note, note_name_to_f
from api.wav_analyzer.wav import SAMPLE_RATE, npFloats, npTuple

matplotlib.use('Agg')

def wav_to_fs(chunk: bytes) -> npTuple:
    data_np = np.frombuffer(chunk, dtype=np.int16)

    samples_no = len(data_np)
    samples_ps = SAMPLE_RATE

    xf = rfftfreq(samples_no, 1 / samples_ps)
    yf = np.abs(rfft(data_np))
    
    max_freq = 4_500 # higher than C8
    i_max: int = np.where(xf < max_freq)[0][-1]

    xf = xf[:i_max]
    yf = yf[:i_max]

    return xf, yf

def fs_to_notes(xf: npFloats, yf: npFloats) -> list[int]:        
    notes: list[int] = list()
    for f in get_maxima(xf, yf):
        notes.append(f_to_note(f))

    return notes


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

def plot_frequencies(xf: npFloats, yf: npFloats, plot_file: str, time_to_delete: float = 5) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude')
        ax.set_xscale('log')
        ax.set_xlim(20, 8000)
        ax.set_ylim(0, 1e7)
        ticks_notes = ['A0', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
        ticks_fs = [note_name_to_f(n) for n in ticks_notes]
        ax.set_xticks(ticks_fs, ticks_notes)
        ax.plot(xf, yf)
        fig.savefig(plot_file)

        def delete_plot_file():
            time.sleep(time_to_delete)
            os.remove(plot_file)

        threading.Thread(target=delete_plot_file).start()
   