import matplotlib.pyplot as plt
import numpy as np
import random


colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

x = np.arange(-5, 5, 0.01)

plt.plot(x, x, label='Linear', c=colors[0])
plt.plot(x, 1.0 / (1.0 + np.exp(-x)), label='Sigmoid', c=colors[1])
plt.plot(x, np.maximum(0, x), label='ReLU', c=colors[2])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.8 * (np.exp(x) - 1)]), label='ELU', c=colors[3])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.1 * x]), label='Leaky ReLU', c=colors[4])

plt.ylim(-1.5, 1.5)

plt.legend()
plt.show()