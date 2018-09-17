import tensorflow as tf
import numpy as np
import resnet
from net_keras import Net
import os
import time
import click
from scipy import io as spio
from callbacks import MomentumScheduler

class Experiment():
	def __init__(self, name, db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum):
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

	def run(self):
		num_channels = 3
		img_size = 32

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

		else:
			raise Exception("Invalid database. Database must be 10, 100 or EMNIST")

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

		if self.net_type == 'resnet56':
			# net = resnet.inference(x, 9, False)
			raise NotImplementedError
		elif self.net_type == 'resnet110':
			# net = resnet.inference(x, 18, False)
			raise NotImplementedError
		elif self.net_type == 'vgg19':
			net_object = Net(img_size, self.activation, num_channels, num_classes, self.spp_alpha)
			model = net_object.vgg19()
		else:
			raise Exception('Invalid net type. You must select one of these: vgg19, resnet56, resnet110')

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

		model.compile(
			optimizer = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.momentum, nesterov=True),
			loss = 'categorical_crossentropy',
			metrics = ['accuracy']
		)

		model.summary()


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