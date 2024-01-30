import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(np.linspace(1, 10, 10), np.linspace(-100, -500, 10))
ax1.set_ylim(-200, 200)
ax1.set_xlim(-200, 200)

ax2.set_ylim(0, 20)
ax2.set_xlim(0, 2000000)
plt.show()
