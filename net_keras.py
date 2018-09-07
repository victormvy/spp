import keras

def parametric_softplus(spp_alpha):
	def spp(x):
		return K.log(1 + K.exp(x)) - spp_alpha
	return spp

class Net:
	def __init__(self, size, activation, num_channels=3, num_classes=5, spp_alpha=0.2):
		self.size = size
		self.activation = activation
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha

		# Add new activation function
		keras.utils.generic_utils.get_custom_objects().update({'spp': keras.layers.Activation(parametric_softplus(spp_alpha))})

	def vgg19(self):
		model = keras.models.Sequential([
			# Block 1
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
								input_shape=(self.size, self.size, self.num_channels), data_format='channels_last'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 2
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 3
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 4
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 5
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Classification block
			keras.layers.Flatten(),
			keras.layers.Dense(4096),
			self.__get_activation(),
			keras.layers.Dense(4096),
			self.__get_activation(),
			keras.layers.Dense(4096, activation='softmax'),

		])

		return model
		
	def __get_activation(self):
		if self.activation == 'relu':
			return keras.layers.Activation('relu')
		elif self.activation == 'lrelu':
			return keras.layers.LeakyReLU()
		elif self.activation == 'prelu':
			return keras.layers.PReLU()
		elif self.activation == 'elu':
			return keras.layers.ELU()
		elif self.activation == 'softplus':
			return keras.layers.Activation('softplus')
		elif self.activation == 'spp':
			return keras.layers.Activation('spp')
		else:
			return keras.layers.Activation('relu')

