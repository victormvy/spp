import keras
from activations import SPP, parametric_softplus, MPELU, RTReLU, RTPReLU, PairedReLU, EReLU, SQRTActivation, CLM, RReLu, PELU, SlopedReLU, PTELU
from layers import GeometricLayer
from resnet import Resnet_2x4

from inception_resnet_v2 import InceptionResNetV2 as Irnv2
from unimodal_extensions import _add_binom_m


class Net:
	def __init__(self, size, activation, final_activation, f_a_params={}, use_tau=True, prob_layer=None, num_channels=3, num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.final_activation = final_activation
		self.f_a_params = f_a_params
		self.use_tau = use_tau
		self.prob_layer = prob_layer
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

		# Add new activation function
		keras.utils.get_custom_objects().update({'spp': SPP(parametric_softplus(spp_alpha))})

	def vgg19(self):
		model = keras.models.Sequential([
			# Block 1
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same',
								   input_shape=(self.size, self.size, self.num_channels), data_format='channels_last'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(64, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 2
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(128, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 3
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(256, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 4
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Block 5
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(512, 3, strides=(1, 1), kernel_initializer='he_uniform', padding='same', ),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			# Classification block
			keras.layers.Flatten(),
			keras.layers.Dropout(rate=self.dropout),
			keras.layers.Dense(4096),
			self.__get_activation(),
			keras.layers.Dense(4096),
			self.__get_activation(),
		])

		model = self.__add_activation(model)

		return model

	def conv128(self):

		feature_filter_size = 3
		classif_filter_size = 4

		model = keras.Sequential([
			keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform', input_shape=(self.size, self.size, self.num_channels),
								   data_format='channels_last'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(32, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(64, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.Conv2D(128, feature_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(),

			keras.layers.Conv2D(128, classif_filter_size, strides=(1, 1),
								   kernel_initializer='he_uniform'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),

			keras.layers.Flatten(),

			keras.layers.Dense(96),

		])

		if self.dropout > 0:
			model.add(keras.layers.Dropout(rate=self.dropout))

		model = self.__add_activation(model)

		return model

	def inception_resnet_v2(self):
		model = keras.Sequential()
		inception = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, input_shape=(self.size, self.size, self.num_channels), classes=self.num_classes)

		# for layer in inception.layers:
		# 	layer.trainable = False

		model.add(inception)
		model.add(keras.layers.Flatten())

		if self.dropout > 0:
			model.add(keras.layers.Dropout(rate=self.dropout))

		model = self.__add_activation(model)

		return model

	def inception_resnet_v2_custom(self):
		input = keras.layers.Input(shape=(self.size, self.size, self.num_channels))
		x = input
		# Required size >= 75 x 75
		size = self.size
		if size < 75:
			size = 75
			x = keras.layers.ZeroPadding2D(padding=(75-self.size) // 2 + 1)(x)

		x = Irnv2(input_tensor=x, include_top=False, input_shape=(size, size, self.num_channels),
						  classes=self.num_classes, pooling='avg', activation=self.__get_activation())(x)

		x = keras.layers.Dense(512)(x)

		if self.dropout > 0:
			x = keras.layers.Dropout(rate=self.dropout)(x)

		x = keras.layers.Dense(self.num_classes)(x)
		x = keras.layers.Activation(self.final_activation)(x)

		model = keras.models.Model(input, x)

		return model

	def beckham_resnet(self):
		resnet = Resnet_2x4((self.size, self.size, self.num_channels), activation=self.activation)
		model = resnet.get_net()

		# model.add(keras.layers.Dense(256))

		if self.dropout > 0:
			model.add(keras.layers.Dropout(rate=self.dropout))

		model = self.__add_activation(model)

		return model


	def testing(self):
		model = keras.Sequential([
			keras.layers.Conv2D(32, 3, strides=(1, 1),
								   kernel_initializer='he_uniform', input_shape=(self.size, self.size, self.num_channels),
								   data_format='channels_last'),
			self.__get_activation(),
			keras.layers.BatchNormalization(),
			keras.layers.MaxPooling2D(pool_size=8),

			keras.layers.Flatten(),
		])

		if self.dropout > 0:
			model.add(keras.layers.Dropout(rate=self.dropout))

		model = self.__add_activation(model)

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
		elif self.activation == 'rrelu':
			return RReLu()
		elif self.activation == 'pelu':
			return PELU()
		elif self.activation == 'slopedrelu':
			return SlopedReLU()
		elif self.activation == 'ptelu':
			return PTELU()
		else:
			return keras.layers.Activation('relu')

	def __add_activation(self, model):
		if self.final_activation == 'poml':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'logit', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'pomp':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'probit', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'pomclog':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'cloglog', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'pomglogit':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'glogit', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'clmcauchit':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'cauchit', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'clmggamma':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'ggamma', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'clmgauss':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'gauss', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'clmexpgauss':
			model.add(keras.layers.Dense(1))
			model.add(keras.layers.BatchNormalization())
			model.add(CLM(self.num_classes, 'expgauss', self.f_a_params, use_tau=self.use_tau))
		elif self.final_activation == 'binomial':
			_add_binom_m(model, self.num_classes, 1.0, 'sigm_learnable')
		else:
			model.add(keras.layers.Dense(self.num_classes))
			if self.prob_layer == 'geometric':
				model.add(GeometricLayer())
			model.add(keras.layers.Activation(self.final_activation))

		return model
