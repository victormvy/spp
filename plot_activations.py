import matplotlib.pyplot as plt
import numpy as np
import random


colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

def randcolor():
	return [random.random(), random.random(), random.random()]

x = np.arange(-5, 5, 0.01)
alpha = 0.08
uniform1 = np.random.uniform(low=1-alpha, high=1+alpha, size=x.size)
uniform2 = np.random.uniform(low=0.45, high=0.55, size=x.size)

# plt.plot(x, x, label='Linear', c=colors[0])
# plt.plot(x, 1.0 / (1.0 + np.exp(-x)), label='Sigmoid', c=colors[1])
# plt.plot(x, np.tanh(x), label='Tanh', c=colors[2])
# plt.plot(x, np.piecewise(x, [x >= 0, x < 0], [lambda x: np.sqrt(x), lambda x: -np.sqrt(-x)]), label='SQRT', c=colors[3])
# plt.plot(x, np.log(1 + np.exp(x)), label='Softplus', c=colors[4])

plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: uniform2[:x.size] * x]), label='RReLU', c=randcolor())
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: uniform1[:x.size] * x, lambda x: 0.5 * x]), label='EPReLU b=0.5', c=colors[6])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: uniform1[:x.size] * x, 0]), label='EReLU', c=colors[5])

plt.plot(x, np.maximum(0, x), label='ReLU', c=colors[0])
plt.plot(x, np.maximum(0, 0.5 * x), label='SlopedReLU a=0.5', c=colors[1])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.5 * x]), label='PReLU a=0.5', c=colors[2])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.8 * (np.exp(x) - 1)]), label='ELU a=0.8', c=colors[3])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.1 * x]), label='Leaky ReLU a=0.1', c=colors[4])
plt.plot(x, np.piecewise(x, [x >= 0, x < 0], [lambda x: (0.5/0.25) * x, lambda x: 0.5 * (np.exp(x/0.25) - 1)]), label='PELU a=0.5 b=0.25', c=randcolor())
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.5 * np.tanh(0.25 * x)]), label='PTELU a=0.5 b=0.25', c=randcolor())


plt.ylim(-2.5, 2.5)

plt.legend()
plt.show()