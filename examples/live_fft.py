import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

CHUNK = 2**15  # 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "./samples/temp.wav"


def print_frequencies(x, y):
    max_amp = max(y)
    i, = np.where(y == max_amp)
    strongest_freq,  = x[i]
    significant_pow = max(y) / 4
    print(f'The significant power threshold is {significant_pow}')
    print("The most significant frequencies:")
    significant_freqs = []
    for i in range(0, len(x)):
        if y[i] > significant_pow:
            significant_freqs.append(x[i])
    print('   '.join(str(v) for v in significant_freqs[:10]))

    # print(' '.join(significant_freqs[:10]))
    print(f'The biggest power is {max_amp} for {strongest_freq} Hz')


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")


def get_one_channel(signal):
    one_channel = []
    for index, datum in enumerate(signal):
        if index % CHANNELS == 0:
            one_channel.append(datum)
    return one_channel


# LOOP for every moment
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    signal = np.frombuffer(data, np.int16)

    moment = get_one_channel(signal)
    frames = len(moment)
    moment_duration = int(frames / RATE * 1000)
    print('Moment duration', moment_duration, 'ms')
    normalized_moment = np.int16((moment / max(moment)) * 32767)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(normalized_moment[:])

    yf = rfft(normalized_moment)
    xf = rfftfreq(frames, 1 / RATE)

    ax2.plot(xf, np.abs(yf))

    print_frequencies(xf, np.abs(yf))


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
