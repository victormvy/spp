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
