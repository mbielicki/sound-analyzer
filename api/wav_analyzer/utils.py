import hashlib
import numpy as np

def save_wav(stream: bytes) -> str:
    filename = f'.\\api\\wav_analyzer\\samples\\{hashlib.sha1(stream).hexdigest()[:8]}.wav'
    file = open(filename, 'wb')
    if not stream[:4] == b'RIFF':
        stream = fix_wav(stream)
    file.write(stream)
    file.close()
    return filename

def fix_wav(stream: bytes) -> bytes:
    headers_file = open('api/wav_analyzer/wav_headers.wav', 'rb')
    headers: bytes = headers_file.read(44)
    
    return headers + stream

def normalize_chunk(chunk: bytes):
    if chunk[:4] == b'RIFF':
        chunk = chunk[44:]

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