import tensorflow as tf
import math


class SPP(tf.keras.layers.Activation):
	"""
	Parametric softplus activation layer.
	"""

	def __init__(self, activation, **kwargs):
		super(SPP, self).__init__(activation, **kwargs)
		self.__name__ = 'SPP'


def parametric_softplus(spp_alpha):
	"""
	Compute parametric softplus function with given alpha.
	:param spp_alpha: alpha parameter for softplus function.
	:return: parametric softplus activation value.
	"""

	def spp(x):
		return tf.log(1 + tf.exp(x)) - spp_alpha

	return spp


class MPELU(tf.keras.layers.Layer):
	def __init__(self, channel_wise=True, **kwargs):
		super(MPELU, self).__init__(**kwargs)
		self.channel_wise = channel_wise

	def build(self, input_shape):
		shape = [1]

		if self.channel_wise:
			shape = [int(input_shape[-1])]  # Number of channels

		self.alpha = self.add_weight(name='alpha', shape=shape, dtype=tf.float32,
									 initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
									 trainable=True)
		self.beta = self.add_weight(name='beta', shape=shape, dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
									trainable=True)

		# Finish buildidng
		super(MPELU, self).build(input_shape)

	def call(self, inputs):
		positive = tf.nn.relu(inputs)
		negative = self.alpha * (tf.exp(-tf.nn.relu(-inputs) * self.beta) - 1)

		return positive + negative

	def compute_output_shape(self, input_shape):
		return input_shape


class RTReLU(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RTReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=tf.float32,
								 initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
								 trainable=False)

		# Finish building
		super(RTReLU, self).build(input_shape)

	def call(self, inputs):
		return tf.nn.relu(inputs + self.a)

	def compute_output_shape(self, input_shape):
		return input_shape


class RTPReLU(tf.keras.layers.PReLU):
	def __init__(self, **kwargs):
		super(RTPReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=tf.float32,
								 initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
								 trainable=False)

		# Call PReLU build method
		super(RTPReLU, self).build(input_shape)

	def call(self, inputs):
		pos = tf.nn.relu(inputs + self.a)
		neg = -self.alpha * tf.nn.relu(-(inputs * self.a))

		return pos + neg


class PairedReLU(tf.keras.layers.Layer):
	def __init__(self, scale=0.5, **kwargs):
		super(PairedReLU, self).__init__(**kwargs)
		self.scale = scale

	def build(self, input_shape):
		self.theta = self.add_weight(name='theta', shape=[1], dtype=tf.float32,
									 initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
									 trainable=True)
		self.theta_p = self.add_weight(name='theta_p', shape=[1], dtype=tf.float32,
									   initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32),
									   trainable=True)

		# Finish building
		super(PairedReLU, self).build(input_shape)

	def call(self, inputs):
		return tf.concat(
			(tf.nn.relu(self.scale * inputs - self.theta), tf.nn.relu(-self.scale * inputs - self.theta_p)),
			axis=len(inputs.get_shape()) - 1)

	def compute_output_shape(self, input_shape):
		return [input_shape[:-1], input_shape[-1] * 2]


class EReLU(tf.keras.layers.Layer):
	def __init__(self, alpha=0.5, **kwargs):
		super(EReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		shape = input_shape[1:]

		self.k = self.add_weight(name='k', shape=shape, dtype=tf.float32,
								 initializer=tf.random_uniform_initializer(minval=1 - self.alpha, maxval=1 + self.alpha,
																		   dtype=tf.float32), trainable=False)

		# Finish building
		super(EReLU, self).build(input_shape)

	def call(self, inputs):
		return tf.nn.relu(inputs * self.k)

	def compute_output_shape(self, input_shape):
		return input_shape


class EPReLU(tf.keras.layers.PReLU):
	def __init__(self, alpha=0.5, **kwargs):
		super(EPReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		shape = input_shape[1:]

		self.k = self.add_weight(name='k', shape=shape, dtype=tf.float32,
								 initializer=tf.random_uniform_initializer(minval=1 - self.alpha, maxval=1 + self.alpha,
																		   dtype=tf.float32), trainable=False)

		# Call PReLU build method
		super(EPReLU, self).build(input_shape)

	def call(self, inputs):
		pos = tf.nn.relu(inputs * self.k)
		neg = -self.alpha * tf.nn.relu(-(inputs))

		return pos + neg


class SQRTActivation(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SQRTActivation, self).__init__(**kwargs)

	def build(self, input_shape):
		super(SQRTActivation, self).build(input_shape)

	def call(self, inputs):
		pos = tf.sqrt(tf.nn.relu(inputs))
		neg = - tf.sqrt(-tf.nn.relu(-inputs))

		return pos + neg


class RReLu(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RReLu, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=input_shape, dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(stddev=1))

		super(RReLu, self).build(input_shape)

	def call(self, inputs):
		pos = tf.nn.relu(inputs)
		neg = self.alpha * tf.nn.relu(-inputs)

		return pos + neg


class PELU(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=tf.float32,
									 initializer=tf.random_uniform_initializer(minval=0.01,
																			   maxval=1,
																			   dtype=tf.float32))
		self.alpha = tf.clip_by_value(self.alpha, 0.0001, 10)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=0.01,
																			  maxval=1,
																			  dtype=tf.float32))
		self.beta = tf.clip_by_value(self.beta, 0.0001, 10)

		super(PELU, self).build(input_shape)

	def call(self, inputs):
		pos = (self.alpha / self.beta) * tf.nn.relu(inputs)
		neg = self.alpha * (tf.exp((-tf.nn.relu(-x)) / self.beta) - 1)

		return pos + neg


class SlopedReLU(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SlopedReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=tf.float32,
									 initializer=tf.random_uniform_initializer(minval=0.01,
																			   maxval=1,
																			   dtype=tf.float32))
		self.alpha = tf.clip_by_value(self.alpha, 0.0001, 10)

		super(SlopedReLU, self).build(input_shape)

	def call(self, inputs):
		return tf.nn.relu(self.alpha * inputs)


class PTELU(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PTELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=tf.float32,
									 initializer=tf.random_uniform_initializer(minval=0.01,
																			   maxval=1,
																			   dtype=tf.float32))
		self.alpha = tf.clip_by_value(self.alpha, 0.0001, 10)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=tf.float32,
									initializer=tf.random_uniform_initializer(minval=0.01,
																			  maxval=1,
																			  dtype=tf.float32))
		self.beta = tf.clip_by_value(self.beta, 0.0001, 10)

		super(PTELU, self).build(input_shape)

	def call(self, inputs):
		pos = tf.nn.relu(inputs)
		neg = self.alpha * tf.tanh(self.beta * tf.nn.relu(-inputs))

		return pos + neg


class NNPOM(tf.keras.layers.Layer):
	"""
	Proportional Odds Model activation layer.
	"""

	def __init__(self, num_classes, link_function, p, use_tau, **kwargs):
		self.num_classes = num_classes
		self.dist = tf.distributions.Normal(loc=0., scale=1.)
		self.link_function = link_function
		self.p = p.copy()
		self.use_tau = use_tau
		super(NNPOM, self).__init__(**kwargs)

	def _convert_thresholds(self, b, a):
		a = tf.pow(a, 2)
		thresholds_param = tf.concat([b, a], axis=0)
		th = tf.reduce_sum(
			tf.matrix_band_part(tf.ones([self.num_classes - 1, self.num_classes - 1]), -1, 0) * tf.reshape(
				tf.tile(thresholds_param, [self.num_classes - 1]), shape=[self.num_classes - 1, self.num_classes - 1]),
			axis=1)
		return th

	def _nnpom(self, projected, thresholds):
		if self.use_tau == 1:
			projected = tf.reshape(projected, shape=[-1]) / self.tau
		else:
			projected = tf.reshape(projected, shape=[-1])

		# projected = tf.Print(projected, data=[tf.reduce_min(projected), tf.reduce_max(projected), tf.reduce_mean(projected)], message='projected min max mean')

		m = tf.shape(projected)[0]
		a = tf.reshape(tf.tile(thresholds, [m]), shape=[m, -1])
		b = tf.transpose(tf.reshape(tf.tile(projected, [self.num_classes - 1]), shape=[-1, m]))
		z3 = a - b

		# z3 = tf.cond(tf.reduce_min(tf.abs(z3)) < 0.01, lambda: tf.Print(z3, data=[tf.reduce_min(tf.abs(z3))], message='z3 abs min', summarize=100), lambda: z3)

		if self.link_function == 'probit':
			a3T = self.dist.cdf(z3)
		elif self.link_function == 'cloglog':
			a3T = 1 - tf.exp(-tf.exp(z3))
		elif self.link_function == 'glogit':
			a3T = 1.0 / tf.pow(1.0 + tf.exp(-self.lmbd * (z3 - self.mu)), self.alpha)
		elif self.link_function == 'cauchit':
			a3T = tf.atan(z3 / math.pi) + 0.5
		elif self.link_function == 'lgamma':
			a3T = tf.cond(self.q < 0, lambda: tf.igammac(tf.pow(self.q, -2), tf.pow(self.q, -2) * tf.exp(self.q * z3)),
						  lambda: tf.cond(self.q > 0, lambda: tf.igamma(tf.pow(self.q, -2),
																		tf.pow(self.q, -2) * tf.exp(self.q * z3)),
										  lambda: self.dist.cdf(z3)))
		elif self.link_function == 'gauss':
			# a3T = 1.0 / 2.0 + tf.sign(z3) * tf.igamma(1.0 / self.alpha, tf.pow(tf.abs(z3) / self.r, self.alpha)) / (2 * tf.exp(tf.lgamma(1.0 / self.alpha)))
			# z3 = tf.Print(z3, data=[tf.reduce_max(tf.abs(z3))], message='z3 abs max')
			a3T = 1.0 / 2.0 + (2 * tf.sigmoid(z3 - self.p['mu']) - 1) * tf.igamma(1.0 / self.p['alpha'],
																	tf.pow(tf.pow((z3 - self.p['mu']) / self.p['r'], 2),
																		   self.p['alpha'])) / (
								  2 * tf.exp(tf.lgamma(1.0 / self.p['alpha'])))
		elif self.link_function == 'expgauss':
			u = self.lmbd * (z3 - self.mu)
			v = self.lmbd * self.sigma
			dist1 = tf.distributions.Normal(loc=0., scale=v)
			dist2 = tf.distributions.Normal(loc=v, scale=tf.pow(v, 2))
			a3T = dist1.cdf(u) - tf.exp(-u + tf.pow(v, 2) / 2 + tf.log(dist2.cdf(u)))
		else:
			a3T = 1.0 / (1.0 + tf.exp(-z3))

		a3 = tf.concat([a3T, tf.ones([m, 1])], axis=1)
		a3 = tf.concat([tf.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, 0:-1]], axis=-1)

		return a3

	def build(self, input_shape):
		self.thresholds_b = self.add_weight('b_b_nnpom', shape=(1,),
											initializer=tf.random_uniform_initializer(minval=0, maxval=0.1))
		self.thresholds_a = self.add_weight('b_a_nnpom', shape=(self.num_classes - 2,),
											initializer=tf.random_uniform_initializer(
												minval=math.sqrt((1.0 / (self.num_classes - 2)) / 2),
												maxval=math.sqrt(1.0 / (self.num_classes - 2))))

		if self.use_tau == 1:
			print('Using tau')
			self.tau = self.add_weight('tau_nnpom', shape=(1,),
									   initializer=tf.random_uniform_initializer(minval=1, maxval=10))
			self.tau = tf.clip_by_value(self.tau, 1, 1000)

		if self.link_function == 'glogit':
			self.lmbd = self.add_weight('lambda_nnpom', shape=(1,),
										initializer=tf.random_uniform_initializer(minval=1, maxval=1))
			self.alpha = self.add_weight('alpha_nnpom', shape=(1,),
										 initializer=tf.random_uniform_initializer(minval=1, maxval=1))
			self.mu = self.add_weight('mu_nnpom', shape=(1,),
									  initializer=tf.random_uniform_initializer(minval=0, maxval=0))
		elif self.link_function == 'lgamma':
			self.q = self.add_weight('q_nnpom', shape=(1,),
									 initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
		elif self.link_function == 'gauss':
			if not 'alpha' in self.p:
				self.p['alpha'] = self.add_weight('alpha_nnpom', shape=(1,), initializer=tf.constant_initializer(0.5))
				self.p['alpha'] = tf.clip_by_value(self.p['alpha'], 0.1, 1.0)

			if not 'r' in self.p:
				self.p['r'] = self.add_weight('r_nnpom', shape=(1,), initializer=tf.constant_initializer(1.0))
				self.p['r'] = tf.clip_by_value(self.p['r'], 0.05, 100)

			if not 'mu' in self.p:
				self.p['mu'] = self.add_weight('mu_nnpom', shape=(1,), initializer=tf.constant_initializer(0.0))

			# self.alpha = self.add_weight('alpha_nnpom', shape=(1,), initializer=tf.constant_initializer(0.3))
			# self.alpha = tf.clip_by_value(self.alpha, 0.2, 0.6)
			# self.alpha = 0.5
			# self.r = self.add_weight('r_nnpom', shape=(1,), initializer=tf.constant_initializer(1.0))
			# self.r = tf.clip_by_value(self.r, 0.2, 100)
			# self.r = 0.3
			# self.mu = self.add_weight('mu_nnpom', shape=(1,), initializer=tf.constant_initializer(0.0))
			# self.mu = 0.0
		elif self.link_function == 'expgauss':
			self.mu = self.add_weight('mu_nnpom', shape=(1,), initializer=tf.constant_initializer(0.0))
			self.sigma = self.add_weight('sigma_nnpom', shape=(1,), initializer=tf.constant_initializer(1.0))
			self.lmbd = self.add_weight('lambda_nnpom', shape=(1,), initializer=tf.constant_initializer(1.0))


	def call(self, x):
		thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
		return self._nnpom(x, thresholds)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)
