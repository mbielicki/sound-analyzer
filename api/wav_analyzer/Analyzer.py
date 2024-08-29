import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hashlib import sha1
from scipy.fft import fft
from .utils import f_to_key, get_maxima

class Analyzer:
    def __init__(self, chunk: bytes):
        self.CHUNK_SIZE = 1024 * 8
        self.CHANNELS = 1
        self.RATE = 48_000 

        self.chunk = chunk
        self.hash = sha1(chunk).hexdigest()[:8]
        self.plot_file = f'.\\api\\wav_analyzer\\samples\\{self.hash}.png'
        self.xf = np.linspace(0, self.RATE, self.CHUNK_SIZE) # frequencies spectrum
        self.yf = self._extract_frequencies()

    def _extract_frequencies(self) -> npt.NDArray[np.float64]:
        data_np = np.frombuffer(self.chunk, dtype='h')

        yf = fft(data_np)
        normalized_yf = np.abs(yf[0:self.CHUNK_SIZE]) / (512 * self.CHUNK_SIZE)
        return normalized_yf
        
    def plot(self):
        plt.xscale('log')
        plt.xlim(28, 4000)
        plt.plot(self.xf, self.yf)
        plt.savefig(self.plot_file)

    def extract_notes(self) -> list[int]:        
        piano_keys: list[int] = list()
        for f in get_maxima(self.xf, self.yf):
            piano_keys.append(f_to_key(f))

        return piano_keys