import threading
import matplotlib.pyplot as plt
import numpy as np
import wave
from play import play
import matplotlib.animation as animation


def animate_whole(filename="output.wav"):
    spf = wave.open(filename, "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    fs = spf.getframerate()

    one_channel = []
    channels_no = spf.getnchannels()
    for index, datum in enumerate(signal):
        if index % channels_no == 0:
            one_channel.append(datum)

    file_duration = len(one_channel) / fs
    Time = np.linspace(0, file_duration, num=len(one_channel))

    fig, ax = plt.subplots()
    line1 = ax.plot(Time, one_channel)[0]
    line2 = ax.plot((0, 0), (-2000, 2000))[0]
    ax.set(xlabel='Time [s]', ylabel='A')

    interval = 20
    frames = int(interval * file_duration)

    def update(frame):
        # update the line plot:
        x = frame / frames * file_duration
        line2.set_xdata((x, x))
        return (line1, line2)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, repeat=False)

    thr = threading.Thread(target=play, args=(), kwargs={})
    thr.start()  # Will run target
    plt.show()
    thr.join()
