import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


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


class QWKCalculation(tf.keras.callbacks.Callback):

	def on_epoch_begin(self, epoch, logs={}):
		# Clear confusion matrix on epoch begin
		self.losses = []


	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


	def on_epoch_end(self, epoch, logs={}):
		pass
		#print(self.losses)

