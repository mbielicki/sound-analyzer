import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CHUNK = 2**11  # 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "./samples/temp.wav"


def get_maxima(x, y):
    maxima = []
    sensible_unit = max(y) / 20
    for i in range(0, len(y)):
        if y[i] > y[i-1] + sensible_unit and y[i] > y[i+1] + sensible_unit:
            maxima.append(x[i])
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


def get_one_channel(signal):
    one_channel = []
    for index, datum in enumerate(signal):
        if index % CHANNELS == 0:
            one_channel.append(datum)
    return one_channel

# LOOP for every moment
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):


def animate(i):
    data = stream.read(CHUNK)
    signal = np.frombuffer(data, np.int16)

    moment = get_one_channel(signal)
    frames = len(moment)
    normalized_moment = np.int16((moment / max(moment)) * 32767)

    ax1.clear()
    ax1.plot(normalized_moment[:])
    ax1.set_ylim(-60_000, 60_000)
    ax1.set_xlim(0, 1050)

    yf = rfft(normalized_moment)
    xf = rfftfreq(frames, 1 / RATE)

    ax2.clear()
    ax2.plot(xf, np.abs(yf))
    ax2.set_ylim(0, 4_000_000)
    ax2.set_xlim(0, 8_000)

    plt.title(get_maxima(xf, np.abs(yf)))


p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


print('Moment duration', int(CHUNK / RATE * 1000), 'ms')
print("* recording")


fig, (ax1, ax2) = plt.subplots(2)
ani = animation.FuncAnimation(fig, animate, interval=50)
plt.show()


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
