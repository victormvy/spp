import tensorflow as tf
from activations import SPP, parametric_softplus, MPELU, RTReLU, RTPReLU, PairedReLU, EReLU, SQRTActivation


class Net:
	def __init__(self, size, activation, num_channels=3, num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

		# Add new activation function
		tf.keras.utils.get_custom_objects().update({'spp': SPP(parametric_softplus(spp_alpha))})

	def vgg19(self):
		model = tf.keras.models.Sequential([
			# Block 1
			tf.keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
								input_shape=(self.size, self.size, self.num_channels), data_format='channels_last'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 2
			tf.keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 3
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 4
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Block 5
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			# Classification block
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dropout(rate=self.dropout),
			tf.keras.layers.Dense(4096),
			self.__get_activation(),
			tf.keras.layers.Dense(4096),
			self.__get_activation(),
			tf.keras.layers.Dense(self.num_classes, activation='softmax'),

		])

		return model

	def conv128(self):

		feature_filter_size = 3
		classif_filter_size = 4

		model = tf.keras.Sequential([
			tf.keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform', input_shape=(128, 128, 3),
								   data_format='channels_last'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			tf.keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			tf.keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			tf.keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			tf.keras.layers.BatchNormalization(),
			tf.keras.layers.MaxPooling2D(),

			tf.keras.layers.Conv2D(128, classif_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),

			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(self.num_classes, activation='softmax'),

		])

		return model
		
	def __get_activation(self):
		if self.activation == 'relu':
			return tf.keras.layers.Activation('relu')
		elif self.activation == 'lrelu':
			return tf.keras.layers.LeakyReLU()
		elif self.activation == 'prelu':
			return tf.keras.layers.PReLU()
		elif self.activation == 'elu':
			return tf.keras.layers.ELU()
		elif self.activation == 'softplus':
			return tf.keras.layers.Activation('softplus')
		elif self.activation == 'spp':
			return tf.keras.layers.Activation('spp')
		elif self.activation == 'mpelu':
			return MPELU(channel_wise=True)
		elif self.activation == 'rtrelu':
			return RTReLU()
		elif self.activation == 'rtprelu':
			return RTPReLU()
		elif self.activation == 'pairedrelu':
			return PairedReLU()
		elif self.activation == 'erelu':
			return EReLU()
		elif self.activation == 'eprelu':
			return EPReLU()
		elif self.activation == 'sqrt':
			return SQRTActivation()
		else:
			return tf.keras.layers.Activation('relu')

