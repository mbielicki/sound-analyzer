from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import numpy as np
import wave
from example_play_plot import animate_whole


def print_frequencies(x, y):
    max_amp = max(y)
    i, = np.where(y == max_amp)
    strongest_freq,  = x[i]
    significant_pow = max(y) / 4
    print(f'The significant power threshold is {significant_pow}')
    print("The most significant frequencies:")
    for i in range(0, len(x)):
        if y[i] > significant_pow:
            print(x[i], end='   ')
    print()
    print(f'The biggest power is {max_amp} for {strongest_freq} Hz')


# Play and show the whole file
filename = "./samples/" + "a1" + "-p.wav"
animate_whole(filename)


# Extract Raw Audio from Wav File
spf = wave.open(filename, "r")
signal = spf.readframes(-1)
signal = np.frombuffer(signal, np.int16)
fs = spf.getframerate()

one_channel = []
channels_no = spf.getnchannels()
for index, datum in enumerate(signal):
    if index % channels_no == 0:
        one_channel.append(datum)

frames = len(one_channel)
file_duration = frames / fs  # seconds
Time = np.linspace(0, file_duration, num=len(one_channel))

# choose moment
moment_start = int(float(input("Moment should start at (seconds): ")
                         ) / file_duration * frames)  # int(frames * 2 / 5)
moment_frames = 5000
moment_fps = 44100  # Hz
moment_duration = int(moment_frames / moment_fps * 1000)
print('Moment duration', moment_duration, 'ms')

moment = one_channel[moment_start: moment_start + moment_frames]
normalized_moment = np.int16((moment / max(moment)) * 32767)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(normalized_moment[:])

yf = rfft(normalized_moment)
xf = rfftfreq(moment_frames, 1 / moment_fps)

ax2.plot(xf, np.abs(yf))

print_frequencies(xf, np.abs(yf))

plt.show()
