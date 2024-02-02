from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.figure import Figure
from lib.piano import piano
from lib.utils_freq_reading import *
import numpy as np
import pyaudio
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

CHUNK = 2**11  # 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "./samples/temp.wav"


p = pyaudio.PyAudio()


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
    global reference_yf
    if i == 0:
        reference_yf = yf

    ax2.clear()
    ax2.plot(xf, np.abs(yf))
    ax2.set_ylim(0, 4_000_000)
    ax2.set_xlim(0, 8_000)

    piano.clear()
    for f in get_maxima(xf, np.abs(yf), reference_yf):
        piano.green(f_to_key(f))


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


print('Moment duration', int(CHUNK / RATE * 1000), 'ms')
print('Moments per second', 1000 / int(CHUNK / RATE * 1000), 'Hz')
print("* recording")


fig, (ax1, ax2) = plt.subplots(2)
ani = animation.FuncAnimation(
    fig, animate, interval=50, cache_frame_data=False)
plt.show()


print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()
