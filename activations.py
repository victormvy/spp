import tensorflow as tf

class SPP(tf.keras.layers.Activation):
	def __init__(self, activation, **kwargs):
		super(SPP, self).__init__(activation, **kwargs)
		self.__name__ = 'SPP'

def parametric_softplus(spp_alpha):
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
			shape = [int(input_shape[-1])] # Number of channels

		self.alpha = self.add_weight(name='alpha', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32), trainable=True)
		self.beta = self.add_weight(name='beta', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32), trainable=True)

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

		self.a = self.add_weight(name='a', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32), trainable=False)

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

		self.a = self.add_weight(name='a', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32), trainable=False)

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
		return tf.concat((tf.nn.relu(self.scale * inputs - self.theta), tf.nn.relu(-self.scale * inputs - self.theta_p)), axis=len(inputs.get_shape()) - 1)

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
		neg = - tf.sqrt(tf.nn.relu(-inputs))

		return pos + neg

class NNPOM(tf.keras.layers.Layer):
	def __init__(self, num_classes, **kwargs):
		self.num_classes = num_classes
		super(NNPOM, self).__init__(**kwargs)

	def _convert_thresholds(self, b, a):
		a = tf.pow(a, 2)
		thresholds_param = tf.concat([b, a], axis=0)
		th = tf.reduce_sum(tf.matrix_band_part(tf.ones([self.num_classes - 1, self.num_classes - 1]), -1, 0) * tf.reshape(
			tf.tile(thresholds_param, [self.num_classes - 1]), shape=[self.num_classes - 1, self.num_classes - 1]), axis=1)
		return th

	def _nnpom(self, projected, thresholds):
		projected = tf.reshape(projected, shape=[-1])
		m = tf.shape(projected)[0]
		a = tf.reshape(tf.tile(thresholds, [m]), shape=[m, -1])
		b = tf.transpose(tf.reshape(tf.tile(projected, [self.num_classes - 1]), shape=[-1, m]))
		z3 = a - b
		a3T = 1.0 / (1.0 + tf.exp(-z3))
		a3 = tf.concat([a3T, tf.ones([m, 1])], axis=1)
		a3 = tf.concat([tf.reshape(a3[:, 0], shape=[-1, 1]), a3[:, 1:] - a3[:, 0:-1]], axis=-1)

		return a3

	def build(self, input_shape):
		self.thresholds_b = self.add_weight('b_b_nnpom', shape=(1,), initializer=tf.random_uniform(shape=(1,), minval=-1, maxval=1))
		self.thresholds_a = self.add_weight('b_a_nnpom', shape=(self.num_classes - 2,),
											initializer=tf.random_uniform(shape=(self.num_classes -2,), minval=-0.5, maxval=0.5))

	def call(self, x):
		thresholds = self._convert_thresholds(self.thresholds_b, self.thresholds_a)
		return self._nnpom(x / 1000.0, thresholds / 1000.0)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], 1)