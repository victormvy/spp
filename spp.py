import tensorflow as tf

class SPP(tf.keras.layers.Activation):
	def __init__(self, activation, **kwargs):
		super(SPP, self).__init__(activation, **kwargs)
		self.__name__ = 'SPP'

def parametric_softplus(spp_alpha):
	def spp(x):
		return tf.log(1 + tf.exp(x)) - spp_alpha
	return spp