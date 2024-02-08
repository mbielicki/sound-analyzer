import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from lib.piano import Piano
from lib.utils_freq_reading import *
import pyaudio
from scipy.fft import fft
import time

CHUNK = 1024 * 8
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

matplotlib.use("TkAgg")

app = tk.Tk()
app.title('Piano')
piano_frame = tk.Frame(app, width=1070, height=120)
plt_frame = tk.Frame(app)

piano = Piano(piano_frame)
piano_frame.pack()

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

chunk_duration = int(CHUNK / RATE * 1000)
print('Chunks duration', chunk_duration, 'ms')
print('Chunks per second', round(1000 / int(CHUNK / RATE * 1000), 2), 'Hz')

# create and pack plots
figure, (ax1, ax2) = plt.subplots(2)

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)       # samples (waveform)
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)

# create a line object with random data
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)
# create semilogx line for spectrum
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

# Signal range is -32k to 32k
# limiting amplitude to +/- 4k
AMPLITUDE_LIMIT = 4096

# format waveform axes
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
ax1.set_xlim(0, 2 * CHUNK)

# format spectrum axes
ax2.set_xlim(20, RATE / 2)

figure_canvas = FigureCanvasTkAgg(figure, plt_frame)
NavigationToolbar2Tk(figure_canvas, plt_frame)
figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
plt_frame.pack()


print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

running = True


def on_tk_close():
    global running
    running = False
    # calculate average frame rate
    frame_rate = frame_count / (time.time() - start_time)

    print('stream stopped')
    print('average frame rate = {:.0f} FPS'.format(frame_rate))
    app.quit()
    app.destroy()


app.protocol('WM_DELETE_WINDOW', on_tk_close)
while running:
    # binary data
    data = stream.read(CHUNK)
    data_np = np.frombuffer(data, dtype='h')

    line.set_ydata(data_np)

    # compute FFT and update line
    yf = fft(data_np)
    normalized_yf = np.abs(yf[0:CHUNK]) / (512 * CHUNK)
    line_fft.set_ydata(normalized_yf)

    piano.clear()
    for f in get_maxima(xf, normalized_yf):
        piano.green(f_to_key(f))

    # update figure canvas
    figure.canvas.draw()
    figure.canvas.flush_events()
    frame_count += 1

stream.stop_stream()
stream.close()
p.terminate()
