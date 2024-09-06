import numpy as np
import numpy.typing as npt

from api.wav_analyzer.notes import note_name_to_n, note_to_f


SAMPLE_RATE = 48_000 

type npFloats = npt.NDArray[np.floating]
type npTuple = tuple[npFloats, npFloats]

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

def remove_wav_headers(chunk: bytes) -> None:
    if chunk[:4] == b'RIFF':
        chunk = chunk[44:]