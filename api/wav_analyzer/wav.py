import numpy as np
import numpy.typing as npt

from api.wav_analyzer.notes import note, note_to_f


SAMPLE_RATE = 48_000 

type npFloats = npt.NDArray[np.floating]
type npTuple = tuple[npFloats, npFloats]

def make_waves(freqs: list[str | float], amps: list[float], duration: float, samples_ps: int = SAMPLE_RATE) -> bytes:
    if len(freqs) != len(amps):
        raise ValueError('freqs and amps must have the same length')
    samples_no = round(duration * samples_ps)
    t = np.linspace(0, duration, samples_no, endpoint=False)
    waves = np.zeros(samples_no)
    for freq, amp in zip(freqs, amps):
        if type(freq) is str:
            freq = note_to_f(note(freq))
        waves += amp * np.sin(2 * np.pi * freq * t)

    chunk = to_wav_bytes(waves)
    return chunk


def to_wav_bytes(waves: npFloats) -> bytes:
    waves = waves * np.iinfo(np.int16).max
    return waves.astype(np.int16).tobytes()

def remove_wav_headers(chunk: bytes) -> None:
    if chunk[:4] == b'RIFF':
        chunk = chunk[44:]