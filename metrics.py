import tensorflow as tf
import numpy as np

def quadratic_weighted_kappa(num_classes, cost_matrix):
	"""
	Compute Quadratic Weighted Kappa metric.
	:param num_classes: number of classes of the dataset.
	:param cost_matrix: cost matrix that will be used for error weights.
	:return: qwk value.
	"""
	def _quadratic_weighted_kappa(y_true, y_pred):
		"""
		Compute Quadratic Weighted Kappa metric.
		:param y_true: true labels.
		:param y_pred: predicted labels.
		:return: qwk value.
		"""
		y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
		y_true = tf.argmax(y_true, 1, output_type=tf.int32)
		conf_mat = tf.confusion_matrix(y_true, y_pred, num_classes=num_classes, dtype=tf.float32)

		hist_y_pred = tf.reshape(
			tf.bincount(y_pred, minlength=num_classes, maxlength=num_classes, dtype=tf.float32),
			shape=[num_classes, 1])
		hist_y_true = tf.reshape(
			tf.bincount(y_true, minlength=num_classes, maxlength=num_classes, dtype=tf.float32),
			shape=[1, num_classes])

		num_scored_items = tf.shape(y_pred)[0]

		expected_count = tf.matmul(hist_y_pred, hist_y_true) / tf.cast(num_scored_items, dtype=tf.float32)

		numerator = tf.reduce_sum(cost_matrix * conf_mat)
		denominator = tf.reduce_sum(cost_matrix * expected_count)

		if denominator == 0:
			return 0

		return 1.0 - numerator / denominator

	return _quadratic_weighted_kappa


def quadratic_weighted_kappa_cm(conf_mat, num_ratings, cost_matrix):
	"""
	Compute QWK function using confusion matrix.

	:param conf_mat: confusion matrix.
	:param min_rating: lowest rating.
	:param max_rating: highest rating.
	:param cost_matrix: cost_matrix.
	:return: QWK value.
	"""
	conf_mat = tf.cast(conf_mat, dtype=tf.float32)

	hist_rater_a = tf.cast(tf.reshape(tf.reduce_sum(conf_mat, axis=1), shape=[num_ratings, 1]),
						   dtype=tf.float32)  # Sum every row
	hist_rater_b = tf.cast(tf.reshape(tf.reduce_sum(conf_mat, axis=0), shape=[1, num_ratings]),
						   dtype=tf.float32)  # Sum every column

	num_scored_items = tf.reduce_sum(conf_mat)  # Sum all the elements

	expected_count = tf.matmul(hist_rater_a, hist_rater_b) / tf.cast(num_scored_items, dtype=tf.float32)

	numerator = tf.reduce_sum(cost_matrix * conf_mat)
	denominator = tf.reduce_sum(cost_matrix * expected_count)

	return 1.0 - numerator / denominator


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Returns the confusion matrix between rater's ratings
	"""
	assert (len(rater_a) == len(rater_b))
	if min_rating is None:
		min_rating = min(rater_a + rater_b)
	if max_rating is None:
		max_rating = max(rater_a + rater_b)
	num_ratings = int(max_rating - min_rating + 1)
	conf_mat = [[0 for i in range(num_ratings)]
				for j in range(num_ratings)]
	for a, b in zip(rater_a, rater_b):
		conf_mat[a - min_rating][b - min_rating] += 1
	return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
	"""
	Returns the counts of each type of rating that a rater made
	"""
	if min_rating is None:
		min_rating = min(ratings)
	if max_rating is None:
		max_rating = max(ratings)
	num_ratings = int(max_rating - min_rating + 1)
	hist_ratings = [0 for x in range(num_ratings)]
	for r in ratings:
		hist_ratings[r - min_rating] += 1

	return hist_ratings

def np_quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
	"""
	Calculates the quadratic weighted kappa
	quadratic_weighted_kappa calculates the quadratic weighted kappa
	value, which is a measure of inter-rater agreement between two raters
	that provide discrete numeric ratings.  Potential values range from -1
	(representing complete disagreement) to 1 (representing complete
	agreement).  A kappa value of 0 is expected if all agreement is due to
	chance.

	quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
	each correspond to a list of integer ratings.  These lists must have the
	same length.

	The ratings should be integers, and it is assumed that they contain
	the complete range of possible ratings.

	quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
	is the minimum possible rating, and max_rating is the maximum possible
	rating
	"""

	# Change values lower than min_rating to min_rating and values
	# higher than max_rating to max_rating.
	rater_a = np.clip(rater_a, min_rating, max_rating)
	rater_b = np.clip(rater_b, min_rating, max_rating)

	rater_a = np.round(rater_a).astype(int).ravel()
	rater_a[~np.isfinite(rater_a)] = 0
	rater_b = np.round(rater_b).astype(int).ravel()
	rater_b[~np.isfinite(rater_b)] = 0

	assert (len(rater_a) == len(rater_b))
	# Get min_rating and max_rating from raters if they are None.
	if min_rating is None:
		min_rating = min(min(rater_a), min(rater_b))
	if max_rating is None:
		max_rating = max(max(rater_a), max(rater_b))
	conf_mat = confusion_matrix(rater_a, rater_b,
								min_rating, max_rating)

	num_ratings = len(conf_mat)
	num_scored_items = float(len(rater_a))

	hist_rater_a = histogram(rater_a, min_rating, max_rating)
	hist_rater_b = histogram(rater_b, min_rating, max_rating)

	numerator = 0.0
	denominator = 0.0

	for i in range(num_ratings):
		for j in range(num_ratings):
			expected_count = (hist_rater_a[i] * hist_rater_b[j]
							  / num_scored_items)
			d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
			numerator += d * conf_mat[i][j] / num_scored_items
			denominator += d * expected_count / num_scored_items

	return 1.0 - numerator / denominator
