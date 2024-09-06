import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import rfft, rfftfreq
import re
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
    notes: list[int] = list()
    for f in get_maxima(xf, yf):
        notes.append(f_to_note(f))

    return notes

def f_to_note(f: float) -> int | None:
    if f < 20:
         return None
    n = round(12 * np.log2(f / 440) + 48)
    if n < 0 or n >= 88:
        return None
    return n

def note_to_f(n: int) -> float:
    return 440 * 2 ** ((n - 48) / 12)

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_n_to_name(n: int) -> str:
    if n < 0:
        raise ValueError(f'note {n} is too low')
    if n >= 88:
         raise ValueError(f'note {n} is too high')
    
    octave = (n - 3) // 12 + 1
    n_in_octave = (n - 3) % 12
    
    return f'{note_names[n_in_octave]}{octave}'

def note_name_to_n(name: str) -> int:
    pattern = r'^[A-Ga-g]#?[0-8]$'
    if re.fullmatch(pattern, name) is None:
        raise ValueError(f'{name} is not a valid note name')
    
    note = name[:-1].upper()

    if note not in note_names:
        raise ValueError(f'{name} is not a valid note name, use one of {note_names}')
    
    octave = int(name[-1])
    n_in_octave = note_names.index(note)

    n = octave * 12 + n_in_octave - 9
    if n < 0 or n >= 88:
        raise ValueError(f'{name} does not fit on a piano keyboard')

    return n
     
type MakeWavesRecipe = list[tuple[float | str, float]]
def make_waves(recipe: MakeWavesRecipe, duration: float, samples_ps: int = SAMPLE_RATE) -> npTuple:
    samples_no = round(duration * samples_ps)
    t = np.linspace(0, duration, samples_no, endpoint=False)
    waves = np.zeros(samples_no)
    for freq, amp in recipe:
        if type(freq) is str:
            freq = note_to_f(note_name_to_n(freq))
        waves += amp * np.sin(2 * np.pi * freq * t)

    return t, waves


def to_wav_bytes(waves: npFloats) -> bytes:
    waves = waves * np.iinfo(np.int16).max
    return waves.astype(np.int16).tobytes()

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