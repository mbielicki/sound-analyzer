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
import wave


class Analyser:
    def __init__(self):
        self.CHUNK = 1024 * 8
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

        self.create_window()
        self.create_plots()

    def create_window(self):
        matplotlib.use("TkAgg")

        self.app = tk.Tk()
        self.app.title('Piano')
        piano_frame = tk.Frame(self.app, width=1070, height=120)
        self.plt_frame = tk.Frame(self.app)

        self.piano = Piano(piano_frame)
        piano_frame.pack()

    def start_recording(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

        chunk_duration = int(self.CHUNK / self.RATE * 1000)
        print('Chunks duration', chunk_duration, 'ms')
        print('Chunks per second', round(
            1000 / int(self.CHUNK / self.RATE * 1000), 2), 'Hz')
        self.recording = True

    async def play(self, filename):
        with wave.open(filename, 'rb') as wf:
            # Instantiate PyAudio and initialize PortAudio system resources (1)
            p = pyaudio.PyAudio()

            # Open stream (2)
            stream = p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            input=True,
                            frames_per_buffer=self.CHUNK)

            # Play samples from the wave file (3)
            while len(data := wf.readframes(self.CHUNK)):  # Requires Python 3.8+ for :=
                stream.write(data)

            stream.close()
            p.terminate()

    def start_reading(self, filename):
        self.recording = False
        wf = wave.open(filename, 'rb')
        self.stream = wf

        def readframes_one_channel(chunk):
            data = self.stream.readframes(int(chunk / wf.getnchannels()))
            if len(data) < self.CHUNK * 2:
                return None
            return data
        self.stream.read = readframes_one_channel
        self.play(filename)

    def create_plots(self):
        # create and pack plots
        self.figure, (ax1, ax2) = plt.subplots(2)

        # variable for plotting
        self.x = np.arange(0, 2 * self.CHUNK, 2)       # samples (waveform)
        # frequencies (spectrum)
        self.xf = np.linspace(0, self.RATE, self.CHUNK)

        # create a line object with random data
        self.line, = ax1.plot(self.x, np.random.rand(self.CHUNK), '-', lw=2)
        # create semilogx line for spectrum
        self.line_fft, = ax2.semilogx(
            self.xf, np.random.rand(self.CHUNK), '-', lw=2)

        # Signal range is -32k to 32k
        # limiting amplitude plot to +/- 4k
        AMPLITUDE_LIMIT = 4096

        # format waveform axes
        ax1.set_title('AUDIO WAVEFORM')
        ax1.set_xlabel('samples')
        ax1.set_ylabel('volume')
        ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
        ax1.set_xlim(0, 2 * self.CHUNK)

        # format spectrum axes
        ax2.set_xlim(20, self.RATE / 2)

        figure_canvas = FigureCanvasTkAgg(self.figure, self.plt_frame)
        NavigationToolbar2Tk(figure_canvas, self.plt_frame)
        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.plt_frame.pack()

    def loop(self):
        # for measuring frame rate
        frame_count = 0
        start_time = time.time()

        self.running = True

        def on_tk_close():
            self.running = False
            # calculate average frame rate
            frame_rate = frame_count / (time.time() - start_time)

            print('stream stopped')
            print('average frame rate = {:.0f} FPS'.format(frame_rate))
            self.app.quit()
            self.app.destroy()

        self.app.protocol('WM_DELETE_WINDOW', on_tk_close)
        while self.running:
            # binary data
            data = self.stream.read(self.CHUNK)
            if not data:
                self.running = False
                break
            data_np = np.frombuffer(data, dtype='h')
            self.line.set_ydata(data_np)

            # compute FFT and update line
            yf = fft(data_np)
            normalized_yf = np.abs(yf[0:self.CHUNK]) / (512 * self.CHUNK)
            self.line_fft.set_ydata(normalized_yf)

            self.piano.clear()
            for f in get_maxima(self.xf, normalized_yf):
                self.piano.green(f_to_key(f))

            # update figure canvas
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            frame_count += 1

        if self.recording:
            self.stream.stop_stream()
        self.stream.close()
        if self.recording:
            self.p.terminate()
