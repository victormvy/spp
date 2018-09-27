import tensorflow as tf

def quadratic_weighted_kappa(num_classes, cost_matrix):
	def quadratic_weighted_kappa(y_true, y_pred):
		y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
		y_true = tf.argmax(y_true, 1, output_type=tf.int32)
		print(y_pred)
		print(y_true)
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

		return 1.0 - numerator / denominator

	return quadratic_weighted_kappa