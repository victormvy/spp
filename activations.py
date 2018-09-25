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
		shape = input_shape[1:]  # Number of channels

		self.k = self.add_weight(name='k', shape=shape, dtype=tf.float32,
								 initializer=tf.random_uniform_initializer(minval=1 - self.alpha, maxval=1 + self.alpha,
																		   dtype=tf.float32), trainable=False)

		# Finish building
		super(EReLU, self).build(input_shape)

	def call(self, inputs):
		return tf.nn.relu(inputs * self.k)

	def compute_output_shape(self, input_shape):
		return input_shape
