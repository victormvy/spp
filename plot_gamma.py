import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import gamma, gammainc


def loggamma(x, a, d, p):
	return gammainc(d/p, (x/a)**p) / gamma(d/p)

plots = [
	{'as': np.round(np.arange(0.1, 1, 0.2), 2), 'ds': np.round(np.arange(0.1, 1, 0.2), 2), 'ps': np.round(np.arange(0.1, 1, 0.2), 2)},
	# {'as': [0.5], 'ds': np.round(np.arange(0.1, 1.5, 0.2), 2), 'ps': [0.1]},
]

for plot in plots:
	aas = plot['as']
	ds = plot['ds']
	ps = plot['ps']

	colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
	for i in range(len(colors), len(aas) * len(ds) * len(ps)):
		colors.append((random.random(), random.random(), random.random()))

	cid = 0
	for a in sorted(aas):
		for d in ds:
			for p in ps:
				x = np.arange(-2, 2, 0.001)
				y = loggamma(x, a, d, p)
				# plt.plot(x,num, linestyle='--', c=colors[cid])
				plt.plot(x, y, label='a={},d={},p={}'.format(a, d, p), c=colors[cid])
				# plt.plot(x,logfunc(x, a, r, mu), label='a={},r={},mu={}'.format(a, r, mu), c = colors[cid])
				cid += 1
	plt.legend()
	plt.show()