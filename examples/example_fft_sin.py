from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100  # Hertz
DURATION = 100 / 1000  # Seconds


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


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
    print(f'The biggest amplitude is {max_amp} for {strongest_freq} Hz')
    significant_amp = get_maxima(x, y)
    print("The most significant frequencies:")
    for i in significant_amp:
        print(i, end='   ')
    print()


_, a_tone = generate_sine_wave(110, SAMPLE_RATE, DURATION)
# _, c_tone = generate_sine_wave(261, SAMPLE_RATE, DURATION)
# _, e_tone = generate_sine_wave(329, SAMPLE_RATE, DURATION)
# _, a4_tone = generate_sine_wave(440, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = a_tone + noise_tone

normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(normalized_tone)

# Number of samples in normalized_tone
N = int(SAMPLE_RATE * DURATION)

yf = rfft(normalized_tone)
xf = rfftfreq(N, 1 / SAMPLE_RATE)

ax2.plot(xf, np.abs(yf))

print_frequencies(xf, np.abs(yf))

plt.show()
