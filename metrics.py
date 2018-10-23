import tensorflow as tf

def quadratic_weighted_kappa(num_classes, cost_matrix):
	def _quadratic_weighted_kappa(y_true, y_pred):
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