import tensorflow as tf

class Net:
	def __init__(self, size, activation, num_channels=3, num_classes=5):
		self.size = size
		self.activation = activation
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = 0

	def vgg19(self, x):
		# Block 1
		net = self.__new_conv_layer(input=x, name= 'block1_conv1', num_input_channels=self.num_channels, filter_size=3, num_filters=64)
		net = self.__new_conv_layer(input=net, name= 'block1_conv2', num_input_channels=64, filter_size=3, num_filters=64)
		net = self.__new_maxpool_2x2(net)
		
		# Block 2
		net = self.__new_conv_layer(input=net, name= 'block2_conv1', num_input_channels=64, filter_size=3, num_filters=128)
		net = self.__new_conv_layer(input=net, name= 'block2_conv2', num_input_channels=128, filter_size=3, num_filters=128)
		net = self.__new_maxpool_2x2(net)
		
		# Block 3
		net = self.__new_conv_layer(input=net, name= 'block3_conv1', num_input_channels=128, filter_size=3, num_filters=256)
		net = self.__new_conv_layer(input=net, name= 'block3_conv2', num_input_channels=256, filter_size=3, num_filters=256)
		net = self.__new_conv_layer(input=net, name= 'block3_conv3', num_input_channels=256, filter_size=3, num_filters=256)
		net = self.__new_conv_layer(input=net, name= 'block3_conv4', num_input_channels=256, filter_size=3, num_filters=256)
		net = self.__new_maxpool_2x2(net)
		
		# Block 4
		net = self.__new_conv_layer(input=net, name= 'block4_conv1', num_input_channels=256, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block4_conv2', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block4_conv3', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block4_conv4', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_maxpool_2x2(net)
		
		# Block 5
		net = self.__new_conv_layer(input=net, name= 'block5_conv1', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block5_conv2', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block5_conv3', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_conv_layer(input=net, name= 'block5_conv4', num_input_channels=512, filter_size=3, num_filters=512)
		net = self.__new_maxpool_2x2(net)
	
		# Classification block
		layer_flat, num_features = self.__flatten_layer(net)
		net = self.__new_fc_layer(name='fc1', input=layer_flat, num_inputs=num_features, num_outputs=4096, use_activation=True)
		net = self.__new_fc_layer(name='fc2', input=net, num_inputs=4096, num_outputs=4096, use_activation=True)
		net = self.__new_fc_layer(name='output_layer', input=net, num_inputs=4096, num_outputs=self.num_classes, use_activation=False)
		
		return net

	# Set new random weights
	def __new_weights(self, name, shape):
		initializer = tf.contrib.layers.variance_scaling_initializer(
			factor=2.0,
			mode='FAN_IN',
			uniform=False
		)

		return tf.get_variable(name, shape=shape, initializer=initializer)

	# Set new biases
	def __new_biases(self, name, length):
		return tf.get_variable(name, initializer=tf.random_uniform(shape=[length], minval=-0.05, maxval=0.05))

	# New convolutional layer (feature layer)
	def __new_conv_layer(self, input, name, num_input_channels, filter_size, num_filters):
		shape = [filter_size, filter_size, num_input_channels, num_filters]
		weights = self.__new_weights("W_" + name, shape)
		biases = self.__new_biases("B_" + name, length=num_filters)

		layer = tf.nn.conv2d(
			input=input,
			filter=weights,
			strides=[1, 1, 1, 1],
			padding='SAME')

		#layer = tf.add(layer, biases)

		layer = self.__activation(layer)

		# Batch normalization reduces training time by reducing internal covariate shift
		mean, var = tf.nn.moments(layer, axes=[0, 1, 2])
		layer = tf.nn.batch_normalization(
			x=layer,
			mean=mean,
			variance=var,
			offset=biases,
			scale=None,
			variance_epsilon=1e-3)

		return layer

	# New pooling layer (reduce dimensionality)
	def __new_maxpool_2x2(self, input):
		layer = tf.nn.max_pool(
			value=input,
			strides=[1, 2, 2, 1],
			ksize=[1, 2, 2, 1],
			padding='SAME')

		return layer

	# New dropout layer (keeps the output of neurons with keep_prob probability)
	def __new_dropout(self, input, keep_prob):
		layer = tf.nn.dropout(x=input, keep_prob=keep_prob)

		return layer

	def __flatten_layer(self, layer):
		layer_shape = layer.get_shape()

		num_features = layer_shape[1:4].num_elements()

		layer_flat = tf.reshape(layer, [-1, num_features])

		return layer_flat, num_features

	def __new_fc_layer(self, input, name, num_inputs, num_outputs, use_activation=True, use_biases=True):
		weights = self.__new_weights(name="W_" + name, shape=[num_inputs, num_outputs])
		if use_biases:
			biases = self.__new_biases(name="B_" + name, length=num_outputs)

		layer = tf.matmul(input, weights)
		if use_biases:
			layer += biases

		if use_activation:
			layer = self.__activation(layer)

		return layer
		
	def __activation(self, x):
		if self.activation == 'relu':
			return tf.nn.relu(x)
		elif self.activation == 'lrelu':
			return tf.nn.leaky_relu(x)
		elif self.activation == 'prelu':
			return self.__parametric_relu(x)
		elif self.activation == 'elu':
			return tf.nn.elu(x)
		elif self.activation == 'softplus':
			return tf.nn.softplus(x)
		elif self.activation == 'spp':
			return self.__parametric_softplus(x)
		else:
			return x
			
	def __parametric_relu(self, _x):
		alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pos = tf.nn.relu(_x)
		neg = alphas * (_x - abs(_x)) * 0.5
		
		return pos + neg
		
	def __parametric_softplus(self, x):
		return tf.log(1 + tf.exp(x)) - self.spp_alpha
