import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import gamma, gammainc


def numerador(x, a, r, mu):
	return gammainc( 1/a, ((((x - mu) / r) ** 2) ** a)  )

def denominador(a):
	return 2 * gamma(1/a)

def logfunc(x, a, r, mu):
	return np.log(np.exp(x) - 1) + np.log(np.exp(x) + 1) + np.log(numerador(x, a, r, mu)) - np.log(2) - np.log(gamma(1.0/a))
 
plots = [
	{'alphas': np.round(np.arange(0.5,1.5,0.2), 2), 'rs':[1.0], 'mus':[0.0]},
	{'alphas': [1.0], 'rs': np.round(np.arange(0.2,1.7,0.2), 2), 'mus':[0.0]},
]


for plot in plots:
	alphas = plot['alphas']
	rs = plot['rs']
	mus = plot['mus']

	colors = ['r','g','b','y','k','b','c','m']
	for i in range(len(colors), len(alphas) * len(rs) * len(mus)):
		colors.append((random.random(), random.random(), random.random()))

	cid = 0
	for a in sorted(alphas):
		for r in rs:
			for mu in mus:
				x = np.arange(-5, 5, 0.001)
				num = numerador(x, a, r, mu)
				total = 0.5 + (2 * (1/(1+np.exp(-(x-mu)))) - 1) * num / denominador(a)
				plt.plot(x,num, linestyle='--', c=colors[cid])
				plt.plot(x,total, label='a={},r={},mu={}'.format(a, r, mu), c=colors[cid])
				# plt.plot(x,logfunc(x, a, r, mu), label='a={},r={},mu={}'.format(a, r, mu), c = colors[cid])
				cid += 1
	plt.legend()
	plt.show()