import tensorflow as tf
import numpy as np


def make_cost_matrix(num_ratings):
	"""
	Create a quadratic cost matrix of num_ratings x num_ratings elements.

	:param thresholds_b: threshold b1.
	:param thresholds_a: thresholds alphas vector.
	:param num_labels: number of labels.
	:return: cost matrix.
	"""

	cost_matrix = np.reshape(np.tile(range(num_ratings), num_ratings), (num_ratings, num_ratings))
	cost_matrix = np.power(cost_matrix - np.transpose(cost_matrix), 2) / (num_ratings - 1) ** 2.0
	return np.float32(cost_matrix)


def qwk_loss(cost_matrix):
	"""
	Compute QWK loss function.

	:param pred_prob: predict probabilities tensor.
	:param true_prob: true probabilities tensor.
	:param cost_matrix: cost matrix.
	:return: QWK loss value.
	"""
	def _qwk_loss(true_prob, pred_prob):
		targets = tf.argmax(true_prob, axis=1)
		costs = tf.gather(cost_matrix, targets)
		# targets = tf.Print(targets, data=[targets], summarize=100, message='targets')

		# costs = tf.Print(costs, data=[costs], summarize=100, message='costs')

#		pred_cls = tf.argmax(pred_prob, axis=1)

# 		conf_mat = tf.confusion_matrix(targets, pred_cls)

		numerator = costs * pred_prob
		numerator = tf.reduce_sum(numerator)

		sum_prob = tf.reduce_sum(pred_prob, axis=0)
		n = tf.reduce_sum(true_prob, axis=0)

		a = tf.reshape(tf.matmul(cost_matrix, tf.reshape(sum_prob, shape=[-1, 1])), shape=[-1])
		b = tf.reshape(n / tf.reduce_sum(n), shape=[-1])

		epsilon = 10e-9

		denominator = a * b
		denominator = tf.reduce_sum(denominator) + epsilon

		return numerator / denominator # + tf.cast(tf.reduce_sum(conf_mat) * 0, dtype=tf.float32)

	return _qwk_loss
