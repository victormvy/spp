import matplotlib.pyplot as plt
import numpy as np
import random
import math
from itertools import repeat

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']

def randcolor():
	return [random.random(), random.random(), random.random()]

x = np.arange(-5, 5, 0.01)

params_dict = {
	'0' : 0.0,
	'0.1' : 0.1,
	'log2' : math.log(2.0),
	'0.5' : 0.5,
	'0.9' : 0.9,
	'1' : 1.0,
	'2' : 2.0
}

lambdas = ['0.1', '0.5', '0.9']
alphas = ['0', 'log2', '1']
betas = ['0', '1', '2']

params = [
	{'lambda' : '0.1', 'alpha' : '0.1', 'beta' : '1'},
	{'lambda' : '0.1', 'alpha' : 'log2', 'beta' : '1'},
	{'lambda' : '0.1', 'alpha' : '0.9', 'beta' : '1'},
	{'lambda' : '0.5', 'alpha' : 'log2', 'beta' : '1'},
	{'lambda' : '0.5', 'alpha' : '0.9', 'beta' : '1'},
	{'lambda' : '0.5', 'alpha' : 'log2', 'beta' : '2'},
	{'lambda' : '0.9', 'alpha' : 'log2', 'beta' : '0'},
	{'lambda' : '0.9', 'alpha' : 'log2', 'beta' : '1'},
	{'lambda' : '0.9', 'alpha' : 'log2', 'beta' : '2'}
]

ELU = lambda x, beta: np.piecewise(x, [x > 0, x <= 0], [lambda x: x, lambda x: beta * (np.exp(x) - 1)])
SPPT = lambda x, alpha: np.log(1 + np.exp(x)) - alpha

for lmbd in lambdas:
	for alpha in alphas:
		for beta in betas:
			plt.plot(x, params_dict[lmbd] * ELU(x, params_dict[beta]) + (1 - params_dict[lmbd]) * SPPT(x, params_dict[alpha]), label=f'λ={lmbd},α={alpha},β={beta}')

# for param in params:
# 	plt.plot(x, params_dict[param['lambda']] * ELU(x, params_dict[param['beta']]) + (1 - params_dict[param['lambda']]) * SPPT(x, params_dict[param['alpha']]), label=f"λ={param['lambda']},α={param['alpha']},β={param['beta']}")



	plt.axhline(0, color='black')
	plt.axvline(0, color='black')
	plt.ylim(-2.5, 2.5)

	plt.legend()
	plt.show()