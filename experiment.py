import tensorflow as tf
import numpy as np
import resnet
from net_keras import Net
import os
import glob
import time
import click
import pickle
import h5py
from scipy import io as spio
from callbacks import MomentumScheduler, ComputeMetricsCallback
from losses import qwk_loss, make_cost_matrix
from metrics import np_quadratic_weighted_kappa, quadratic_weighted_kappa_cm
from dataset import Dataset
from sklearn.metrics import confusion_matrix
from math import inf


class Experiment():
	"""
	Class that represents a single experiment that can be run and evaluated.
	"""
	def __init__(self, name='unnamed', db='100', net_type='vgg19', batch_size=128, epochs=100,
				 checkpoint_dir='checkpoint', loss='crossentropy', activation='relu', final_activation='softmax',
				 prob_layer=None,
				 spp_alpha=1.0, lr=0.1, momentum=0.9, dropout=0, task='both', workers=4, val_metrics=['loss', 'acc']):
		self._name = name
		self._db = db
		self._net_type = net_type
		self._batch_size = batch_size
		self._epochs = epochs
		self._checkpoint_dir = checkpoint_dir
		self._loss = loss
		self._activation = activation
		self._final_activation = final_activation
		self._prob_layer = prob_layer
		self._spp_alpha = spp_alpha
		self._lr = lr
		self._momentum = momentum
		self._dropout = dropout
		self._task = task
		self._finished = False
		self._workers = workers
		self._val_metrics = val_metrics

		self._best_metric = inf

	def set_auto_name(self):
		"""
		Set experiment name based on experiment parameters.
		:return: None
		"""
		self.name = self.get_auto_name()

	def get_auto_name(self):
		"""
		Get experiment auto-generated name based on experiment parameters.
		:return: experiment auto-generated name.
		"""
		return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.db, self.net_type, self.batch_size, self.activation,
														 self.loss,
														 self.final_activation,
														 self.prob_layer and self.prob_layer or '',
														 self.spp_alpha, self.lr,
														 self.momentum, self.dropout)

	# PROPERTIES

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		self._name = name

	@name.deleter
	def name(self):
		del self._name

	@property
	def db(self):
		return self._db

	@db.setter
	def db(self, db):
		self._db = db

	@db.deleter
	def db(self):
		del self._db

	@property
	def net_type(self):
		return self._net_type

	@net_type.setter
	def net_type(self, net_type):
		self._net_type = net_type

	@net_type.deleter
	def net_type(self):
		del self._net_type

	@property
	def batch_size(self):
		return self._batch_size

	@batch_size.setter
	def batch_size(self, batch_size):
		self._batch_size = batch_size

	@batch_size.deleter
	def batch_size(self):
		del self._batch_size

	@property
	def epochs(self):
		return self._epochs

	@epochs.setter
	def epochs(self, epochs):
		self._epochs = epochs

	@epochs.deleter
	def epochs(self):
		del self._epochs

	@property
	def checkpoint_dir(self):
		return self._checkpoint_dir

	@checkpoint_dir.setter
	def checkpoint_dir(self, checkpoint_dir):
		self._checkpoint_dir = checkpoint_dir

	@checkpoint_dir.deleter
	def checkpoint_dir(self):
		del self._checkpoint_dir

	@property
	def loss(self):
		return self._loss

	@loss.setter
	def loss(self, loss):
		self._loss = loss

	@loss.deleter
	def loss(self):
		del self._loss

	@property
	def activation(self):
		return self._activation

	@activation.setter
	def activation(self, activation):
		self._activation = activation

	@activation.deleter
	def activation(self):
		del self._activation

	@property
	def final_activation(self):
		return self._final_activation

	@final_activation.setter
	def final_activation(self, final_activation):
		self._final_activation = final_activation

	@final_activation.deleter
	def final_activation(self):
		del self._final_activation

	@property
	def prob_layer(self):
		return self._prob_layer

	@prob_layer.setter
	def prob_layer(self, prob_layer):
		self._prob_layer = prob_layer

	@prob_layer.deleter
	def prob_layer(self):
		del self._prob_layer

	@property
	def spp_alpha(self):
		return self._spp_alpha

	@spp_alpha.setter
	def spp_alpha(self, spp_alpha):
		self._spp_alpha = spp_alpha

	@spp_alpha.deleter
	def spp_alpha(self):
		del self._spp_alpha

	@property
	def lr(self):
		return self._lr

	@lr.setter
	def lr(self, lr):
		self._lr = lr

	@lr.deleter
	def lr(self):
		del self._lr

	@property
	def momentum(self):
		return self._momentum

	@momentum.setter
	def momentum(self, momentum):
		self._momentum = momentum

	@momentum.deleter
	def momentum(self):
		del self._momentum

	@property
	def dropout(self):
		return self._dropout

	@dropout.setter
	def dropout(self, dropout):
		self._dropout = dropout

	@dropout.deleter
	def dropout(self):
		del self._dropout

	@property
	def task(self):
		return self._task

	@task.setter
	def task(self, task):
		self._task = task

	@task.deleter
	def task(self):
		del self._task

	@property
	def finished(self):
		return self._finished

	@finished.setter
	def finished(self, finished):
		self._finished = finished

	@finished.deleter
	def finished(self):
		del self._finished

	@property
	def workers(self):
		return self._workers

	@workers.setter
	def workers(self, workers):
		self._workers = workers

	@workers.deleter
	def workers(self):
		del self._workers

	@property
	def val_metrics(self):
		return self._val_metrics

	@val_metrics.setter
	def val_metrics(self, val_metrics):
		self._val_metrics = val_metrics

	@val_metrics.deleter
	def val_metrics(self):
		del self._val_metrics

	@property
	def best_metric(self):
		return self._best_metric

	def new_metric(self, metric):
		"""
		Updates best metric if metric provided is better than the best metric stored.
		:param metric: new metric.
		:return: True if new metric is better than best metric or False otherwise.
		"""
		if metric <= self._best_metric:
			self._best_metric = metric
			return True
		return False

	# # # # # # #

	def run(self):
		"""
		Run training process.
		:return: None
		"""

		print('=== RUNNING {} ==='.format(self.name))

		# Train data generator
		train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			rescale=1. / 255,
			# shear_range=0.2,
			# zoom_range=0.2,
			# horizontal_flip=True,
			# vertical_flip=True,
			# brightness_range=(0.5, 1.5),
			# rotation_range=90,
			# fill_mode='nearest'
		)

		# Validation data generator
		val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

		# Get database paths
		train_path, val_path, _ = self.get_db_path(self.db)

		# Check that database exists and paths are correct
		if train_path == '' or val_path == '':
			raise Exception('Invalid database. Choose one of: Retinopathy or Adience.')

		# Load datasets
		ds_train = Dataset(train_path)
		ds_val = Dataset(val_path)

		# Get dataset details
		num_classes = ds_train.num_classes
		num_channels = ds_train.num_channels
		img_size = ds_train.img_size

		# Train data generator used for training
		train_generator = train_datagen.flow(
			ds_train.x,
			ds_train.y,
			batch_size=self.batch_size
		)

		# Validation generator
		val_generator = val_datagen.flow(
			ds_val.x,
			ds_val.y,
			batch_size=self.batch_size,
			shuffle=False
		)

		# Calculate the number of steps per epoch
		steps = (len(ds_train.y) * 1) // self.batch_size

		# Get class weights based on frequency
		class_weight = ds_train.get_class_weights()

		# Learning rate scheduler callback
		def learning_rate_scheduler(epoch):
			return self.lr * np.exp(-0.025 * epoch)

		# Save epoch callback for training process
		def save_epoch(epoch, logs):
			# Check whether new metric is better than best metric
			if (self.new_metric(logs['val_loss'])):
				model.save(os.path.join(self.checkpoint_dir, best_model_file))

			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'w') as f:
				f.write(str(epoch + 1))
				f.write('\n' + str(self.best_metric))

		save_epoch_callback = tf.keras.callbacks.LambdaCallback(
			on_epoch_end=save_epoch
		)

		# NNet object
		net_object = Net(img_size, self.activation, self.final_activation, self.prob_layer, num_channels, num_classes,
						 self.spp_alpha,
						 self.dropout)

		model = self.get_model(net_object, self.net_type)

		# Create checkpoint dir if not exists
		if not os.path.isdir(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		# Model and results file names
		model_file = 'model.h5'
		best_model_file = 'best_model.h5'
		model_file_extra = 'model.txt'
		csv_file = 'results.csv'

		# Initial epoch. 0 by default
		start_epoch = 0

		# Check whether a saved model exists
		if os.path.isfile(os.path.join(self.checkpoint_dir, model_file)) and os.path.isfile(
				os.path.join(self.checkpoint_dir, model_file_extra)):
			print("===== RESTORING SAVED MODEL =====")
			model.load_weights(os.path.join(self.checkpoint_dir, model_file))

			# Continue from the epoch where we were and load the best metric
			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'r') as f:
				start_epoch = int(f.readline())
				self.new_metric(float(f.readline()))

		# Create the cost matrix that will be used to compute qwk
		cost_matrix = tf.constant(make_cost_matrix(num_classes), dtype=tf.float32)

		# Cross-entropy loss by default
		loss = 'categorical_crossentropy'

		# Quadratic Weighted Kappa loss
		if self.loss == 'qwk':
			loss = qwk_loss(cost_matrix)

		# Only accuracy for training.
		# Computing QWK for training properly is too expensive
		metrics = ['accuracy']

		# Compile the keras model
		model.compile(
			optimizer=tf.keras.optimizers.Adam(lr=self.lr),
			loss=loss,
			metrics=metrics
		)

		# Print model summary
		model.summary()

		# Run training
		model.fit_generator(train_generator, epochs=self.epochs,
							initial_epoch=start_epoch,
							steps_per_epoch=steps,
							callbacks=[#tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
										tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=6, mode='min', min_lr=1e-4, verbose=1),
									   ComputeMetricsCallback(num_classes, val_generator=val_generator,
															  val_batches=ds_val.num_batches(self.batch_size),
															  metrics=self.val_metrics),
									   tf.keras.callbacks.ModelCheckpoint(
										   os.path.join(self.checkpoint_dir, model_file)),
									   save_epoch_callback,
									   tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, csv_file),
																	append=True),
									   tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir),
									   ],
							workers=self.workers,
							use_multiprocessing=True,
							max_queue_size=self.batch_size * 3,
							class_weight=class_weight
							)

		self.finished = True

	def evaluate(self):
		"""
		Run evaluation on test data.
		:return: None
		"""
		print('=== EVALUATING {} ==='.format(self.name))

		evaluation_file = 'evaluation.txt'

		if os.path.isfile(os.path.join(self.checkpoint_dir, evaluation_file)):
			with open(os.path.join(self.checkpoint_dir, evaluation_file), 'r') as f:
				metric = float(f.readline())
				print('Metric found in file: {} (Evaluation skipped)'.format(metric))
				return metric

		_, _, test_path = self.get_db_path(self.db)

		# Load test dataset
		ds_test = Dataset(test_path)

		# Get dataset details
		num_classes = ds_test.num_classes
		num_channels = ds_test.num_channels
		img_size = ds_test.img_size

		# Validation data generator
		test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

		# Validation generator
		test_generator = test_datagen.flow(
			ds_test.x,
			ds_test.y,
			batch_size=self.batch_size,
			shuffle=False
		)

		# NNet object
		net_object = Net(img_size, self.activation, self.final_activation, self.prob_layer, num_channels, num_classes,
						 self.spp_alpha,
						 self.dropout)

		model = self.get_model(net_object, self.net_type)

		best_model_file = 'best_model.h5'

		# Check if best model file exists
		if not os.path.isfile(os.path.join(self.checkpoint_dir, best_model_file)):
			print('Best model file not found')
			return

		# Restore weights
		model.load_weights(os.path.join(self.checkpoint_dir, best_model_file))

		# Create the cost matrix that will be used to compute qwk
		cost_matrix = tf.constant(make_cost_matrix(num_classes), dtype=tf.float32)

		# Cross-entropy loss by default
		loss = 'categorical_crossentropy'

		# Quadratic Weighted Kappa loss
		if self.loss == 'qwk':
			loss = qwk_loss(cost_matrix)

		# Only accuracy for training.
		# Computing QWK for training properly is too expensive
		metrics = ['accuracy']

		# Compile the keras model
		model.compile(
			optimizer=tf.keras.optimizers.Adam(lr=self.lr),
			loss=loss,
			metrics=metrics
		)

		# Get predictions
		test_generator.reset()
		predictions = model.predict_generator(
			test_generator
		)

		# Calculate metric
		metric = np_quadratic_weighted_kappa(np.argmax(ds_test.y, axis=1), np.argmax(predictions, axis=1), 0, num_classes-1)

		with open(os.path.join(self.checkpoint_dir, evaluation_file), 'w') as f:
			f.write(str(metric))

		return metric

	def get_model(self, net_object, name):
		if name == 'vgg19':
			model = net_object.vgg19()
		elif name == 'conv128':
			model = net_object.conv128()
		elif name == 'testing':
			model = net_object.testing()
		elif name == 'inceptionresnetv2':
			model = net_object.inception_resnet_v2_custom()
		elif name == 'beckhamresnet':
			model = net_object.beckham_resnet()
		else:
			raise Exception('Invalid net type. You must select one of these: vgg19, conv128')

		return model

	def get_db_path(self, db):
		"""
		Get dataset path for train, validation and test for a given database name.
		:param db: database name.
		:return: train path, validation path, test path.
		"""
		if db.lower() == 'retinopathy':
			return "../retinopathy/128/train", "../retinopathy/128/val", "../retinopathy/128/test"
		elif db.lower() == 'retinopathy256':
			return "../retinopathy/256/train", "../retinopathy/256/val", "../retinopathy/256/test"
		elif db.lower() == 'adience':
			return "../adience/256/train", "../adience/256/val", "../adience/256/test"
		elif db.lower() == 'cifar10' or db.lower() == 'cifar100':
			return db.lower() + 'train', db.lower() + 'val', db.lower() + 'test'
		else:
			return "", "", ""

	def get_config(self):
		"""
		Get config dictionary from object config.
		:return: config dictionary.
		"""
		return {
			'name': self.name,
			'db': self.db,
			'net_type': self.net_type,
			'batch_size': self.batch_size,
			'epochs': self.epochs,
			'checkpoint_dir': self.checkpoint_dir,
			'prob_layer': self.prob_layer,
			'loss': self.loss,
			'activation': self.activation,
			'final_activation': self.final_activation,
			'spp_alpha': self.spp_alpha,
			'lr': self.lr,
			'momentum': self.momentum,
			'dropout': self.dropout,
			'task': self.task,
			'workers' : self.workers,
			'val_metrics' : self.val_metrics
		}

	def set_config(self, config):
		"""
		Set object config from config dictionary
		:param config: config dictionary.
		:return: None
		"""
		self.db = 'db' in config and config['db'] or '10'
		self.net_type = 'net_type' in config and config['net_type'] or 'vgg19'
		self.batch_size = 'batch_size' in config and config['batch_size'] or 128
		self.epochs = 'epochs' in config and config['epochs'] or 100
		self.checkpoint_dir = 'checkpoint_dir' in config and config['checkpoint_dir'] or 'results'
		self.loss = 'loss' in config and config['loss'] or 'crossentropy'
		self.activation = 'activation' in config and config['activation'] or 'relu'
		self.final_activation = 'final_activation' in config and config['final_activation'] or 'softmax'
		self.prob_layer = 'prob_layer' in config and config['prob_layer'] or None
		self.spp_alpha = 'spp_alpha' in config and config['spp_alpha'] or 0
		self.lr = 'lr' in config and config['lr'] or 0.1
		self.momentum = 'momentum' in config and config['momentum'] or 0
		self.dropout = 'dropout' in config and config['dropout'] or 0
		self.task = 'task' in config and config['task'] or 'both'
		self.workers = 'workers' in config and config['workers'] or 4
		self.val_metrics = 'val_metrics' in config and config['val_metrics'] or ['acc', 'loss']

		if 'name' in config:
			self.name = config['name']
		else:
			self.set_auto_name()

	def save_to_file(self, path):
		"""
		Save experiment to pickle file.
		:param path: path where pickle file will be saved.
		:return: None
		"""
		pickle.dump(self.get_config(), path)

	def load_from_file(self, path):
		"""
		Load experiment from pickle file.
		:param path: path where pickle file is located.
		:return: None
		"""
		if os.path.isfile(path):
			self.set_config(pickle.load(path))
