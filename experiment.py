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
from callbacks import MomentumScheduler, QWKCalculation
from losses import qwk_loss, make_cost_matrix
from metrics import quadratic_weighted_kappa

class Experiment():
	def __init__(self, name='unnamed', db='100', net_type='vgg19', batch_size=128, epochs=100, checkpoint_dir='checkpoint', activation='relu', spp_alpha=1.0, lr=0.1, momentum=0.9, dropout=0):
		self._name = name
		self._db = db
		self._net_type = net_type
		self._batch_size = batch_size
		self._epochs = epochs
		self._checkpoint_dir = checkpoint_dir
		self._activation = activation
		self._spp_alpha = spp_alpha
		self._lr = lr
		self._momentum = momentum
		self._dropout = dropout
		self._finished = False

	def set_auto_name(self):
		self.name = self.get_auto_name()

	def get_auto_name(self):
		return "{}_{}_{}_{}_{}_{}_{}".format(self.db, self.net_type, self.batch_size, self.activation, self.spp_alpha, self.lr,
													  self.momentum)

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
	def activation(self):
		return self._activation

	@activation.setter
	def activation(self, activation):
		self._activation = activation

	@activation.deleter
	def activation(self):
		del self._activation

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
		train_dataset = None
		test_dataset = None

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
			train_path = "../retinopathy_small/tiny/train"
			test_path = "../retinopathy_small/tiny/val"
			num_classes = 5
			num_channels = 3
			img_size = 128
			img_shape = (img_size, img_size, num_channels)

			train_filenames = []
			for file in glob.glob(os.path.join(train_path, "*.tfrecords")):
				train_filenames.append(file)

			test_filenames = []
			for file in glob.glob(os.path.join(test_path, "*.tfrecords")):
				test_filenames.append(file)

			if len(train_filenames) == 0 or len(test_filenames) == 0:
				raise Exception('Invalid database')

			def parser(record):
				"""
				Read samples from record and decode them as image-label.
				This function is used with dataset.map() function

				:param record: record from tfrecords file.
				:return: image, label.
				"""

				keys_to_features = {
					"image": tf.FixedLenFeature((), tf.string, default_value=""),
					"label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
				}
				parsed = tf.parse_single_example(record, keys_to_features)

				# Perform additional preprocessing on the parsed data.
				image = tf.decode_raw(parsed["image"], tf.int64)
				image = tf.reshape(image, img_shape)
				image = (tf.cast(image, tf.float32) / 255.0) * 2.0 - 1.0
				label = tf.cast(parsed["label"], tf.int32)
				label = tf.one_hot(label, depth=num_classes, dtype=tf.float32)

				return image, label

			train_dataset = tf.data.Dataset.from_tensor_slices(train_filenames) \
				.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
							cycle_length=len(train_filenames), block_length=1)\
				.shuffle(buffer_size=10000, reshuffle_each_iteration=False)\
				.repeat() \
				.apply(tf.contrib.data.map_and_batch(parser, self.batch_size, num_parallel_calls=8)) \
				.prefetch(tf.contrib.data.AUTOTUNE)

			test_dataset = tf.data.TFRecordDataset(test_filenames, compression_type='GZIP')\
				.repeat()\
				.apply(tf.contrib.data.map_and_batch(parser, self.batch_size, num_parallel_calls=8))\
				.prefetch(tf.contrib.data.AUTOTUNE)

		else:
			raise Exception('Invalid database. Choose one of: 10, 100, EMNIST or Retinopathy.')


		if not train_x is None and not test_x is None and not train_y_cls is None and not test_y_cls is None:
			train_x = train_x / 255.0
			test_x = test_x / 255.0

			train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls), num_classes])
			test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls), num_classes])

		def learning_rate_scheduler(epoch):
			final_lr = self.lr
			if epoch >= 60:
				final_lr -= 0.02
			if epoch >= 80:
				final_lr -= 0.02
			if epoch >= 90:
				final_lr -= 0.02
			return float(final_lr)

		def momentum_scheduler(epoch):
			final_mmt = self.momentum
			if epoch >= 60:
				final_mmt -= 0.0005
			if epoch >= 80:
				final_mmt -= 0.0005
			if epoch >= 90:
				final_mmt -= 0.0005
			return float(final_mmt)

		def save_epoch(epoch, logs):
			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'w') as f:
				f.write(str(epoch + 1))

		save_epoch_callback = tf.keras.callbacks.LambdaCallback(
			on_epoch_end=save_epoch
		)

		net_object = Net(img_size, self.activation, num_channels, num_classes, self.spp_alpha, self._dropout)
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

		model_file = 'model.h5py'
		model_file_extra = 'model.txt'
		csv_file = 'results.csv'

		start_epoch = 0

		if os.path.isfile(os.path.join(self.checkpoint_dir, model_file)) and os.path.isfile(os.path.join(self.checkpoint_dir, model_file_extra)):
			print("===== RESTORING SAVED MODEL =====")
			model.load_weights(os.path.join(self.checkpoint_dir, model_file))

			with open(os.path.join(self.checkpoint_dir, model_file_extra), 'r') as f:
				start_epoch = int(f.readline())


		cost_matrix = tf.constant(make_cost_matrix(num_classes), dtype=tf.float32)

		model.compile(
			optimizer = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True),
			loss =  qwk_loss(cost_matrix), # 'categorical_crossentropy',
			metrics = ['accuracy', quadratic_weighted_kappa(num_classes, cost_matrix)]
		)

		model.summary()

		if not train_x is None and not test_x is None and not train_y is None and not test_y is None:
			model.fit(x=train_x, y=train_y, batch_size=self.batch_size, epochs=self.epochs, initial_epoch=start_epoch,
					  callbacks=[ tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
								  MomentumScheduler(momentum_scheduler),
								  tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_dir, model_file)),
								  save_epoch_callback,
								  tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, csv_file), append=True),
								  tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir)
								  ],
					  validation_data=(test_x, test_y)
					  )
		elif train_dataset and test_dataset:
			model.fit(x=train_dataset.make_one_shot_iterator(), y=None, batch_size=None, epochs=self.epochs, initial_epoch=start_epoch,
					  steps_per_epoch=100000//self.batch_size,
					  callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
								 MomentumScheduler(momentum_scheduler),
								 tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_dir, model_file)),
								 save_epoch_callback,
								 tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_dir, csv_file), append=True),
								 tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_dir),
								 ],
					  validation_data=test_dataset.make_one_shot_iterator(),
					  validation_steps=3525 // self.batch_size
					  )
		else:
			raise Exception('Database not initialized')

		self.finished = True


	def get_config(self):
		return {
			'name' : self.name,
			'db' : self.db,
			'net_type' : self.net_type,
			'batch_size' : self.batch_size,
			'epochs' : self.epochs,
			'checkpoint_dir' : self.checkpoint_dir,
			'activation' : self.activation,
			'spp_alpha' : self.spp_alpha,
			'lr' : self.lr,
			'momentum' : self.momentum,
			'dropout' : self.dropout
		}

	def set_config(self, config):
		self.db = config['db']
		self.net_type = config['net_type']
		self.batch_size = config['batch_size']
		self.epochs = config['epochs']
		self.checkpoint_dir = config['checkpoint_dir']
		self.activation = config['activation']
		self.spp_alpha = config['spp_alpha']
		self.lr = config['lr']
		self.momentum = config['momentum']
		self.dropout = config['dropout']

		if 'name' in config:
			self.name = config['name']
		else:
			self.set_auto_name()

	def save_to_file(self, path):
		pickle.dump(self.get_config(), path)

	def load_from_file(self, path):
		if os.file.exists(path):
			self.set_config(pickle.load(path))