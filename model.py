import tensorflow as tf
import functools
import tflearn


def lazy_property(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator


class VGG19Model():
	def __init__(self, image, label, num_classes, activation):
		self.image = image
		self.label = label
		self.num_classes = num_classes
		self.__activation = activation
		self.prediction
		self.optimize
		self.error

	@lazy_property
	def prediction(self):
		x = self.image

		x = tf.contrib.layers.repeat(x, 2, tf.contrib.layers.conv2d, 64, 3, activation_fn=self.activation,
									 scope='block1_conv')
		x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME', scope='block1_maxpool')

		x = tf.contrib.layers.repeat(x, 2, tf.contrib.layers.conv2d, 128, 3, activation_fn=self.activation,
									 scope='block2_conv')
		x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME', scope='block2_maxpool')

		x = tf.contrib.layers.repeat(x, 4, tf.contrib.layers.conv2d, 256, 3, activation_fn=self.activation,
									 scope='block3_conv')
		x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME', scope='block3_maxpool')

		x = tf.contrib.layers.repeat(x, 4, tf.contrib.layers.conv2d, 512, 3, activation_fn=self.activation,
									 scope='block4_conv')
		x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME', scope='block4_maxpool')

		x = tf.contrib.layers.repeat(x, 4, tf.contrib.layers.conv2d, 512, 3, activation_fn=self.activation,
									 scope='block5_conv')
		x = tf.contrib.layers.max_pool2d(x, 2, padding='SAME', scope='block5_maxpool')

		x = tf.contrib.layers.repeat(x, 2, tf.contrib.layers.fully_connected, 4096, activation_func=self.activation, scope='fc')
		x = tf.contrib.layers.fully_connected(x, self.num_classes, activation_func=None)

		return x

	@lazy_property
	def optimize(self):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label)
		cost = tf.reduce_mean(cross_entropy)
		optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.1, use_nesterov=True).minimize(cost)
		return optimizer

	@lazy_property
	def activation(self):
		if self.__activation == 'lrelu':
			return tf.nn.leaky_relu
		elif self.__activation == 'prelu':
			return tflearn.activation.prelu
		elif self.__activation == 'elu':
			return tf.nn.elu
		elif self.__activation == 'softplus':
			return tf.nn.softplus
		elif self.__activation == 'spp':
			return self.__parametric_softplus
		else:
			return tf.nn.relu

	def __parametric_softplus(self, x):
		return tf.log(1 + tf.exp(x)) - self.spp_alpha
