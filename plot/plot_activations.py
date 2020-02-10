import matplotlib.pyplot as plt
import numpy as np
import random
import colorsys

# colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']



def randcolor():
	return [random.random(), random.random(), random.random()]

colors = np.array([(255, 0, 0),(255, 162, 0),(255, 221, 0),(0, 0, 0),(4, 255, 0),(0, 184, 129),(0, 225, 255),(0, 128, 255),(43, 0, 255),
		  (153, 0, 255),(255, 0, 234),(255, 0, 115),(110, 0, 0),(110, 86, 0),(53, 110, 0),(0, 110, 105),(0, 39, 110),(92, 0, 110)])
colors = colors / 255.0

x = np.arange(-5, 5, 0.01)
alpha = 0.08
uniform1 = np.random.uniform(low=1-alpha, high=1+alpha, size=x.size)
uniform2 = np.random.uniform(low=0.45, high=0.55, size=x.size)
random1 = np.random.normal(loc=0.0, scale=.1, size=x.size)

# plt.plot(x, x, label='Linear', c=colors[0])
# plt.plot(x, 1.0 / (1.0 + np.exp(-x)), label='Sigmoid', c=colors[1])
# plt.plot(x, np.tanh(x), label='Tanh', c=colors[2])

plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.8 * (np.exp(x) - 1)]), label='ELU {alpha=0.8}', c=colors[0])
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: uniform1[:x.size] * x, lambda x: 0.5 * x]), label='EPReLU {beta=0.50}', c=colors[1], linestyle='--')
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: uniform1[:x.size] * x, 0]), label='EReLU {no params}', c=colors[2])
plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.1 * x]), label='LReLU {alpha=0.1}', c=colors[4])
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.5 * (np.exp(0.5 * x) - 1)]), label='MPELU {alpha=0.50, beta=0.5}', c=colors[4], linestyle='--')
# c = colors[5]
# plt.plot(x, np.maximum(-0.5 * x + 0.5, 0), label='PairedReLU {s=0.30, s_p=-0.50, theta=0.20, theta_p=-0.50}', c=c, linestyle='--')
# plt.plot(x, np.maximum(0.3 * x - 0.2, 0), c=c, linestyle='--')
# plt.plot(x, np.piecewise(x, [x >= 0, x < 0], [lambda x: (0.5/0.25) * x, lambda x: 0.5 * (np.exp(x/0.25) - 1)]), label='PELU {alpha=0.50, beta=0.25}', c=colors[6], linestyle='--')
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.5 * x]), label='PReLU {alpha=0.50}', c=colors[7], linestyle='--')
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: 0.5 * np.tanh(0.25 * x)]), label='PTELU {alpha=0.50, beta=0.25}', c=colors[8], linestyle='--')
plt.plot(x, np.maximum(0, x), label='ReLU {no params}', c=colors[9])
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: uniform2[:x.size] * x]), label='RReLU {no params}', c=colors[10])
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x + random1[:x.size], lambda x: 0.5 * (x + random1[:x.size])]), label='RTPReLU {alpha=0.50}', c=colors[11], linestyle='--')
# plt.plot(x, np.piecewise(x, [x > 0, x <= 0], [lambda x: x + random1[:x.size], lambda x: 0]), label='RTReLU {alpha=0.50}', c=colors[12])
# plt.plot(x, np.maximum(0, 0.5 * x), label='SlopedReLU {alpha=0.50}', c=colors[13], linestyle='--')
# plt.plot(x, np.piecewise(x, [x >= 0, x < 0], [lambda x: np.sqrt(x), lambda x: -np.sqrt(-x)]), label='SQRT {no params}', c=colors[14])
# plt.plot(x, np.log(1 + np.exp(x)), label='Softplus {no params}', c=colors[15])


plt.axhline(0, color='grey', linestyle='--', linewidth=0.4)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.4)

plt.ylim(-2.5, 2.5)

plt.legend()
plt.show()