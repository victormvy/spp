from losses import emd_loss
from keras import backend as K
from pyemd import emd
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance
import keras

def tril_indices(n, k=0):
	"""Return the indices for the lower-triangle of an (n, m) array.
	Works similarly to `np.tril_indices`
	Args:
	  n: the row dimension of the arrays for which the returned indices will
		be valid.
	  k: optional diagonal offset (see `np.tril` for details).
	Returns:
	  inds: The indices for the triangle. The returned tuple contains two arrays,
		each with the indices along one dimension of the array.
	"""
	m1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
	m2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
	mask = (m1 - m2) >= -k
	ix1 = tf.boolean_mask(m2, tf.transpose(mask))
	ix2 = tf.boolean_mask(m1, tf.transpose(mask))
	return ix1, ix2


def ecdf(p):
	"""Estimate the cumulative distribution function.
	The e.c.d.f. (empirical cumulative distribution function) F_n is a step
	function with jump 1/n at each observation (possibly with multiple jumps
	at one place if there are ties).
	For observations x= (x_1, x_2, ... x_n), F_n is the fraction of
	observations less or equal to t, i.e.,
	F_n(t) = #{x_i <= t} / n = 1/n \sum^{N}_{i=1} Indicator(x_i <= t).
	Args:
	  p: a 2-D `Tensor` of observations of shape [batch_size, num_classes].
		Classes are assumed to be ordered.
	Returns:
	  A 2-D `Tensor` of estimated ECDFs.
	"""
	n = p.get_shape().as_list()[1]
	indices = tril_indices(n)
	indices = tf.transpose(tf.stack([indices[1], indices[0]]))
	ones = tf.ones([n * (n + 1) / 2])
	triang = tf.scatter_nd(indices, ones, [n, n])
	return tf.matmul(p, triang)


def emd_loss2(p, p_hat, r=2, scope=None):
	"""Compute the Earth Mover's Distance loss.
	Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
	Distance-based Loss for Training Deep Neural Networks." arXiv preprint
	arXiv:1611.05916 (2016).
	Args:
	  p: a 2-D `Tensor` of the ground truth probability mass functions.
	  p_hat: a 2-D `Tensor` of the estimated p.m.f.-s
	  r: a constant for the r-norm.
	  scope: optional name scope.
	`p` and `p_hat` are assumed to have equal mass as \sum^{N}_{i=1} p_i =
	\sum^{N}_{i=1} p_hat_i
	Returns:
	  A 0-D `Tensor` of r-normed EMD loss.
	"""
	with tf.name_scope(scope, 'EmdLoss', [p, p_hat]):
		ecdf_p = ecdf(p)
		ecdf_p_hat = ecdf(p_hat)
		emd = tf.reduce_mean(tf.pow(tf.abs(ecdf_p - ecdf_p_hat), r), axis=-1)
		# emd = tf.pow(emd, 1 / r)
		return tf.reduce_mean(emd)


def np_emd_loss(true_prob, pred_prob, num_classes):
	targets = np.argmax(true_prob, axis=1)
	A = 0
	B = 0

	for x, k in zip(pred_prob, targets):
		for i in range(0, k-1):
			temp = 0
			for j in range(0, i):
				temp += x[j]
			A += pow(temp, 2)

		for i in range(k, num_classes):
			temp = 0
			for j in range(0, i):
				temp += x[j] - 1
			B += pow(temp, 2)

	return (A + B) / true_prob.shape[0]

def emd_loss3(true, pred):
	sum = 0

	for t, p in zip(true, pred):
		sum += wasserstein_distance(t, p)

	return sum / true.shape[0]


def np_matrix_emd_loss(true_prob, pred_prob):
	true_cumsum = np.cumsum(true_prob, axis=1)
	pred_cumsum = np.cumsum(pred_prob, axis=1)

	return np.mean(np.square(pred_cumsum - true_cumsum))




# true = K.cast([[1,0,0],[1,0,0],[0,1,0],[0,0,1]], K.floatx())
# pred = [[0.6,0.3,0.1],[0.7,0.8,0.4],[0.2,0.6,0.4],[0.3,0.5,0.9]]
# true_ = [[1,0,0],[0,1,0],[0,0,1]]
# true = K.cast([[1,0,0],[0,1,0],[0,0,1]], K.floatx())
# pred_ = [[0.6,0.3,0.1],[0.2,0.7,0.4],[0.1,0.1,0.45]]
# pred = K.cast([[0.6,0.3,0.1],[0.2,0.7,0.4],[0.1,0.1,0.45]], K.floatx())

# true = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0], [0.0,0.0,1.0], [0.0,0.0,1.0]])
# pred = np.array([[0.6,0.3,0.1],[0.2,0.7,0.1],[0.25,0.3,0.45], [0.2,0.7,0.1],[0.25,0.3,0.45], [0.2,0.7,0.1]])

num_classes = 3
true = keras.utils.to_categorical(np.random.randint(num_classes, size=(30)))
pred = np.random.rand(30, num_classes)
pred = pred / pred.sum(axis=1)[:,np.newaxis]

print(true.shape)
print(pred.shape)

sess = K.get_session()


loss_res = sess.run(emd_loss(K.cast(true, K.floatx()), K.cast(pred, K.floatx())))
print("TF", loss_res)

print('Numpy Matrix', np_matrix_emd_loss(true, pred))

print('Numpy', np_emd_loss(true, pred, num_classes))

print('TF github', sess.run(emd_loss2(K.cast(true, K.floatx()), K.cast(pred, K.floatx()))))

# print(emd(true, pred, np.array([[0.0, 0.0], [0.0, 1.0]])))

print('scipy', emd_loss3(true, pred))