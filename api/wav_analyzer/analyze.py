import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import fft
from .utils import f_to_key, get_maxima

import os
import time
import threading

matplotlib.use('Agg')

CHUNK_SIZE = 1024 * 8
CHANNELS = 1
RATE = 48_000 

xf = np.linspace(0, RATE, CHUNK_SIZE) # frequencies spectrum

def extract_frequencies(chunk: bytes) -> npt.NDArray[np.float64]:
        data_np = np.frombuffer(chunk, dtype='h')

        yf = fft(data_np)
        normalized_yf = np.abs(yf[0:CHUNK_SIZE]) / (512 * CHUNK_SIZE)
        return normalized_yf

def extract_notes(yf: npt.NDArray[np.float64]) -> list[int]:        
        piano_keys: list[int] = list()
        for f in get_maxima(xf, yf):
            piano_keys.append(f_to_key(f))

        return piano_keys

def plot_frequencies(yf: npt.NDArray[np.float64], plot_file: str):
        fig, ax = plt.subplots()
        ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude')
        ax.set_xscale('log')
        ax.set_xlim(28, 4000)
        ax.set_ylim(0, 10)
        ax.plot(xf, yf)
        fig.savefig(plot_file)

        def delete_plot_file():
            time.sleep(5)
            os.remove(plot_file)

        threading.Thread(target=delete_plot_file).start()