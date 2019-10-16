import keras
from keras import backend as K

class GeometricLayer(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(GeometricLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.num_classes = input_shape[1]

		super(GeometricLayer, self).build(input_shape)

	def call(self, inputs):
		return K.pow(1. - inputs, int(self.num_classes)) * inputs


class DenseMultiplicative(keras.layers.Dense):
	def build(self, input_shape):
		print(input_shape)

		super(DenseMultiplicative, self).build(input_shape)

	def call(self, inputs):
		# Tile inputs vector (units x inputs)
		t_inputs = K.tile(inputs, (self.units, 1))

		# inputs pow kernel (weights)
		inputs_pow = K.pow(t_inputs, self.kernel)

		# Product of columns
		out = K.prod(inputs_pow, axis=1)

		return out