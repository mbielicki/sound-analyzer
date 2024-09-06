import re
import numpy as np

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

def note_name(n: int) -> str:
    if n < 0:
        raise ValueError(f'note {n} is too low')
    if n >= 88:
         raise ValueError(f'note {n} is too high')
    
    octave = (n - 3) // 12 + 1
    n_in_octave = (n - 3) % 12
    
    return f'{note_names[n_in_octave]}{octave}'

def note(name: str) -> int:
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
     