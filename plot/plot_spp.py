import matplotlib.pyplot as plt
import numpy as np
import random
import math
from itertools import repeat

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

def randcolor():
	return [random.random(), random.random(), random.random()]

x = np.arange(-5, 5, 0.01)
# alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
alphas = [0.0, math.log(2), 1.0]
alpha_label = ['0.0', 'log2', '1.0']


for alpha, lbl in zip(alphas, alpha_label):
	plt.plot(x, np.log(1 + np.exp(x)) - alpha, label='α=' + lbl)

# Asymptotes
plt.plot(x, x, linestyle='dashed', color='gray')
plt.text(-1.82,-1.77, 'y=x', horizontalalignment='right')
plt.plot(x, x - math.log(2), linestyle='dashed', color='gray')
plt.text(-1.42,-2.1, 'y=x - log2', horizontalalignment='right')
plt.plot(x, x - 1, linestyle='dashed', color='gray')
plt.text(-1.2, -2.4, 'y=x - 1')

plt.plot(x, np.repeat(- math.log(2), x.size), linestyle='dashed', color='gray')
plt.text(3, 0.1 -math.log(2), 'y=-log2')
plt.plot(x, np.repeat(-1, x.size), linestyle='dashed', color='gray')
plt.text(3,-0.92, 'y=-1')

plt.axhline(0, color='black')
plt.axvline(0, color='black')






plt.ylim(-2.5, 2.5)

plt.legend()
plt.show()