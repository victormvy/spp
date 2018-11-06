import tensorflow as tf
import numpy as np
import resnet
from net_keras import Net
import os
import glob
import time
import click
import pickle
from scipy import io as spio
from callbacks import MomentumScheduler, ComputeMetricsCallback
from losses import qwk_loss, make_cost_matrix
from metrics import quadratic_weighted_kappa
from dataset import Dataset


class Experiment():
	def __init__(self, name='unnamed', db='100', net_type='vgg19', batch_size=128, epochs=100,
				 checkpoint_dir='checkpoint', loss='crossentropy', activation='relu', final_activation='softmax',
				 prob_layer=None,
				 spp_alpha=1.0, lr=0.1, momentum=0.9, dropout=0):
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
		self._finished = False

		self._best_qwk = -1

	def set_auto_name(self):
		self.name = self.get_auto_name()

	def get_auto_name(self):
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
	def finished(self):
		return self._finished

	@finished.setter
	def finished(self, finished):
		self._finished = finished

	@finished.deleter
	def finished(self):
		del self._finished

	@property
	def best_qwk(self):
		return self._best_qwk

	def new_qwk(self, qwk):
		if qwk >= self._best_qwk:
			self._best_qwk = qwk
			return True
		return False

	# # # # # # #

	def run(self):
		print('=== RUNNING {} ==='.format(self.name))

		num_channels = 3
		img_size = 32

		train_x = None
		test_x = None
		train_y_cls = None
		test_y_cls = None
		train_y = None
		test_y = None

		class_weight = None

		train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			vertical_flip=True,
			brightness_range=(0.5, 1.5),
			rotation_range=90,
			fill_mode='nearest'
		)

		test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

		train_generator = None
		test_generator = None

		if self.db == '10':
			(train_x, train_y_cls), (test_x, test_y_cls) = tf.keras.datasets.cifar10.load_data()
			num_classes = 10
		elif self.db == '100':
			(train_x, train_y_cls), (test_x, test_y_cls) = tf.keras.datasets.cifar100.load_data()
			num_classes = 100
		elif self.db.lower() == 'emnist':
			emnist = spio.loadmat('emnist/emnist-byclass.mat')

			train_x = np.reshape(emnist['dataset'][0][0][0][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
			train_y_cls = emnist['dataset'][0][0][0][0][0][1]

			test_x = np.reshape(emnist['dataset'][0][0][1][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
			test_y_cls = emnist['dataset'][0][0][1][0][0][1]

			num_classes = 62
			num_channels = 1
			img_size = 28
		elif self.db.lower() == 'retinopathy':
			train_path = "../retinopathy/128/train"
			test_path = "../retinopathy/128/val"
			num_classes = 5
			num_channels = 3
			img_size = 128
			img_shape = (img_size, img_size, num_channels)

			ds_train = Dataset(train_path)
			ds_test = Dataset(test_path)

			train_generator = train_datagen.flow(
				ds_train.x,
				ds_train.y,
				# target_size=(img_size, img_size),
				batch_size=self.batch_size,
				# class_mode='categorical'
			)

			test_generator = test_datagen.flow(
				ds_test.x,
				ds_test.y,
				# target_size=(img_size, img_size),
				batch_size=self.batch_size,
				# class_mode='categorical'
			)

			class_weight = {
				0: 1,
				1: 3.4,
				2: 3.15,
				3: 3.59,
				4: 3.63
			}


		elif self.db.lower() == 'adience':
			train_path = "../adience/"
			test_path = "../adience/"
			num_channels = 3
			img_size = 140

			ds_train = Dataset(train_path)
			ds_test = Dataset(test_path)

			num_classes = ds_test.num_classes

			train_generator = train_datagen.flow(
				ds_train.x,
				ds_train.y,
				# target_size=(img_size, img_size),
				batch_size=self.batch_size,
				# class_mode='categorical'
			)

			test_generator = test_datagen.flow(
				ds_test.x,
				ds_test.y,
				# target_size=(img_size, img_size),
				batch_size=self.batch_size,
				# class_mode='categorical'
			)

			class_weight = ds_train.get_class_weights()

		else:
			raise Exception('Invalid database. Choose one of: 10, 100, EMNIST or Retinopathy.')

		if not train_x is None and not test_x is None and not train_y_cls is None and not test_y_cls is None:
			train_x = train_x / 255.0
			test_x = test_x / 255.0

			train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls), num_classes])
			test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls), num_classes])

		def learning_rate_scheduler(epoch):
			return self.lr * np.exp(-0.025 * epoch)
		# return self.lr / (1 + epoch / 30)

		def save_epoch(epoch, logs):
			# Check whether new qwk is better than best qwk
			if (self.new_qwk(logs['val_qwk'])):
				model.save(os.path.join(self.checkpoint_dir, best_model_file))

			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'w') as f:
				f.write(str(epoch + 1))
				f.write('\n' + str(self.best_qwk))

		save_epoch_callback = tf.keras.callbacks.LambdaCallback(
			on_epoch_end=save_epoch
		)

		net_object = Net(img_size, self.activation, self.final_activation, self.prob_layer, num_channels, num_classes,
						 self.spp_alpha,
						 self.dropout)
		if self.net_type == 'resnet56':
			# net = resnet.inference(x, 9, False)
			raise NotImplementedError
		elif self.net_type == 'resnet110':
			# net = resnet.inference(x, 18, False)
			raise NotImplementedError
		elif self.net_type == 'vgg19':
			model = net_object.vgg19()
		elif self.net_type == 'conv128':
			model = net_object.conv128()
		else:
			raise Exception('Invalid net type. You must select one of these: vgg19, resnet56, resnet110, conv128')

		if not os.path.isdir(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		model_file = 'model.hdf5'
		best_model_file = 'best_model.hdf5'
		model_file_extra = 'model.txt'
		csv_file = 'results.csv'

		start_epoch = 0

		if os.path.isfile(os.path.join(self.checkpoint_dir, model_file)) and os.path.isfile(
				os.path.join(self.checkpoint_dir, model_file_extra)):
			print("===== RESTORING SAVED MODEL =====")
			model.load_weights(os.path.join(self.checkpoint_dir, model_file))

			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'r') as f:
				start_epoch = int(f.readline())
				self.new_qwk(float(f.readline()))

		cost_matrix = tf.constant(make_cost_matrix(num_classes), dtype=tf.float32)

		loss = 'categorical_crossentropy'

		if self.loss == 'qwk':
			loss = qwk_loss(cost_matrix)

		metrics = ['accuracy']
		# If database is retinopathy, add qwk metric
		# if self.db == 'retinopathy':
		# 	metrics.append(quadratic_weighted_kappa(num_classes, cost_matrix))

		model.compile(
			optimizer=tf.keras.optimizers.Adam(lr=self.lr),
			# tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True),
			loss=loss,
			metrics=metrics
		)

		model.summary()

		if not train_x is None and not test_x is None and not train_y is None and not test_y is None:
			model.fit(x=train_x, y=train_y, batch_size=self.batch_size, epochs=self.epochs, initial_epoch=start_epoch,
					  callbacks=[  # tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
						  # MomentumScheduler(momentum_scheduler),
						  tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_dir, model_file)),
						  save_epoch_callback,
						  tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, csv_file), append=True),
						  tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir)
					  ],
					  validation_data=(test_x, test_y)
					  )
		elif train_generator and test_generator:
			model.fit_generator(train_generator, epochs=self.epochs,
								initial_epoch=start_epoch,
								steps_per_epoch=100000 // self.batch_size,
								callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
										   # MomentumScheduler(momentum_scheduler),
										   ComputeMetricsCallback(num_classes, val_generator=test_generator,
																  val_batches=ds_test.num_batches(self.batch_size)),
										   tf.keras.callbacks.ModelCheckpoint(
											   os.path.join(self.checkpoint_dir, model_file)),
										   save_epoch_callback,
										   tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, csv_file),
																		append=True),
										   tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir),
										   ],
								workers=4,
								use_multiprocessing=True,
								max_queue_size=self.batch_size * 20,
								class_weight=class_weight
								)
		else:
			raise Exception('Database not initialized')

		self.finished = True

	def get_config(self):
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
			'dropout': self.dropout
		}

	def set_config(self, config):
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

		if 'name' in config:
			self.name = config['name']
		else:
			self.set_auto_name()

	def save_to_file(self, path):
		pickle.dump(self.get_config(), path)

	def load_from_file(self, path):
		if os.path.isfile(path):
			self.set_config(pickle.load(path))
