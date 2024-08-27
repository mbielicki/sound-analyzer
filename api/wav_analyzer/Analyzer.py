import numpy as np
from scipy.fft import fft
from .utils import f_to_key, get_maxima, normalize_chunk

class Analyzer:
    def __init__(self):
        self.CHUNK = 1024 * 8
        self.CHANNELS = 1
        self.RATE = 44100 #48000 

        self.xf = np.linspace(0, self.RATE, self.CHUNK) # frequencies spectrum
        
    def extract_notes(self, chunk: bytes) -> list[int]:
        normalize_chunk(chunk)

        data_np = np.frombuffer(chunk, dtype='h')

        # compute FFT and update line
        yf = fft(data_np)
        normalized_yf = np.abs(yf[0:self.CHUNK]) / (512 * self.CHUNK)

        piano_keys: list[int] = list()
        for f in get_maxima(self.xf, normalized_yf):
            piano_keys.append(f_to_key(f))

        return piano_keys
    

