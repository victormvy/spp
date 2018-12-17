import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from metrics import quadratic_weighted_kappa_cm
from losses import make_cost_matrix

class MomentumScheduler(tf.keras.callbacks.Callback):
	'''Momentum scheduler.
	# Arguments
	schedule: a function that takes an epoch index (integer, indexed from 0) and current momentum as input
	and returns a new momentum as output (float).
	'''
	def __init__(self, schedule):
		super(MomentumScheduler, self).__init__()
		self.schedule = schedule

	def on_epoch_begin(self, epoch, logs={}):
		assert hasattr(self.model.optimizer, 'momentum'), \
		'Optimizer must have a "momentum" attribute.'
		mmtm = self.schedule(epoch)
		assert type(mmtm) == float, 'The output of the "schedule" function should be float.'
		tf.assign(self.model.optimizer.momentum, mmtm)


class ComputeMetricsCallback(tf.keras.callbacks.Callback):
	"""
	Callback that computes train and/or validation metrics for a batch.
	Computed metrics are: accuracy, loss and QWK.
	"""

	def __init__(self, num_classes, train_generator=None, val_generator=None, train_batches=None, val_batches=None, metrics=['loss', 'acc']):
		self.train_generator = train_generator
		self.val_generator = val_generator
		self.train_batches = train_batches
		self.val_batches = val_batches
		self.metrics = metrics
		self.classes = []
		self.num_classes = num_classes
		self.cost_matrix = make_cost_matrix(self.num_classes)

		for i in range(0, num_classes):
			self.classes.append(i)

	def _compute_metrics(self, generator, num_batches, classes):
		sess = tf.keras.backend.get_session()
		conf_mat = None
		mean_acc = 0
		mean_loss = 0
		batch_count = 0

		for x, y in generator:
			if batch_count >= num_batches:
				break

			prediction = self.model.predict_on_batch(x)
			loss = self.model.test_on_batch(x, y)[0]
			y = np.argmax(y, axis=1)
			prediction = np.argmax(prediction, axis=1)

			if conf_mat is None:
				conf_mat = confusion_matrix(y, prediction, labels=classes)
			else:
				conf_mat += confusion_matrix(y, prediction, labels=classes)

			batch_count += 1
			mean_acc += accuracy_score(y, prediction)
			mean_loss += loss

		mean_acc /= batch_count
		mean_loss /= batch_count
		qwk = sess.run(quadratic_weighted_kappa_cm(conf_mat, self.num_classes, self.cost_matrix))

		metrics = {}

		if 'acc' in self.metrics:
			metrics['acc'] = mean_acc
		if 'loss' in self.metrics:
			metrics['loss'] = mean_loss
		if 'qwk' in self.metrics:
			metrics['qwk'] = qwk

		metrics['conf_mat'] = conf_mat

		return metrics

	def on_epoch_end(self, epoch, logs={}):
		if self.train_generator and self.train_batches:
			train_metrics = self._compute_metrics(self.train_generator, self.train_batches, self.classes)

			s = '\n'

			if 'acc' in self.metrics:
				logs['train_acc'] = train_metrics['acc']
				s += 'train_acc: {} '.format(train_metrics['acc'])
			if 'qwk' in self.metrics:
				logs['train_qwk'] = train_metrics['qwk']
				s += 'train_qwk: {} '.format(train_metrics['qwk'])
			if 'loss' in self.metrics:
				logs['train_loss'] = train_metrics['loss']
				s += 'train_loss: {}'.format(train_metrics['loss'])

			print(s)

			print('TRAIN CONF MATRIX')
			print(train_metrics['conf_mat'])

		if self.val_generator and self.val_batches:
			val_metrics = self._compute_metrics(self.val_generator, self.val_batches, self.classes)

			s = '\n'

			if 'acc' in self.metrics:
				logs['val_acc'] = val_metrics['acc']
				s += 'val_acc: {} '.format(val_metrics['acc'])
			if 'qwk' in self.metrics:
				logs['val_qwk'] = val_metrics['qwk']
				s += 'val_qwk: {} '.format(val_metrics['qwk'])
			if 'loss' in self.metrics:
				logs['val_loss'] = val_metrics['loss']
				s += 'val_loss: {}'.format(val_metrics['loss'])

			print(s)

			print('VALIDATION CONF MATRIX')
			print(val_metrics['conf_mat'])