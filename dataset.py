import imageio
import numpy as np
import os
import math
import h5py
from sklearn.model_selection import train_test_split
import keras
import cv2
import pandas as pd
from skimage import io
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight


class Dataset:
	"""
	Class that represents a dataset that is loaded from a file.
	"""
	def __init__(self, path="", portion=1.0):
		self._data = {'x' : [], 'y' : []}
		self._num_classes = 0
		self._portion = portion

		if path != "":
			self.load(path)

		super(Dataset, self).__init__()

	@property
	def x(self):
		return np.array(self._data['x'])

	@x.setter
	def x(self, x):
		self._data['x'] = x

	@x.deleter
	def x(self):
		del self._data['x']

	@property
	def y(self):
		return np.array(self._data['y'])

	@y.setter
	def y(self, y):
		self._data['y'] = y

	@y.deleter
	def y(self):
		del self._data['y']

	@property
	def data(self):
		return np.array(self._data)

	@data.setter
	def data(self, data):
		self._data = data

	@data.deleter
	def data(self):
		del self._data

	@property
	def num_classes(self):
		return self._num_classes

	@num_classes.setter
	def num_classes(self, num_classes):
		self._num_classes = num_classes

	@num_classes.deleter
	def num_classes(self):
		del self._num_classes

	@property
	def sample_shape(self):
		return self._sample_shape

	@sample_shape.setter
	def sample_shape(self, sample_shape):
		self._sample_shape = sample_shape

	@sample_shape.deleter
	def sample_shape(self):
		del self._sample_shape

	@property
	def portion(self):
		return self._portion

	@portion.setter
	def portion(self, portion):
		self._portion = portion

	@portion.deleter
	def portion(self):
		del self._portion

	@property
	def mean(self):
		return np.array(self._data['x']).mean()

	@property
	def std(self):
		return np.array(self._data['x']).std()

	def load(self, path):
		if path == 'cifar10train':
			self._load_cifar10('train')
		elif path == 'cifar10val':
			self._load_cifar10('val')
		elif path == 'cifar10test':
			self._load_cifar10('test')
		elif path == 'mnisttrain':
			self._load_mnist('train')
		elif path == 'mnistval':
			self._load_mnist('val')
		elif path == 'mnisttest':
			self._load_mnist('test')
		elif path == 'cinic10train':
			self._load_cinic10('train')
		elif path == 'cinic10val':
			self._load_cinic10('val')
		elif path == 'cinic10test':
			self._load_cinic10('test')
		elif os.path.isdir(path):
			self._load_from_dir(path)
		else:
			self._load_from_h5(path)

	def _load_from_dir(self, path):
		"""
		Load dataset from directory.
		There should be one subdirectory for each class.
		:param path: dataset path.
		:return: None
		"""
		self._data = {}
		self._data['x'] = []
		self._data['y'] = []
		self._num_classes = len(os.listdir(path))
		self._sample_shape = None
		for cls in os.listdir(path):
			flist = os.listdir(os.path.join(path, cls))
			for i, f in enumerate(flist):
				# Take just a portion of the data
				if i > len(flist) * self.portion:
					break

				file_path = os.path.join(os.path.join(path, cls), f)
				if os.path.isfile(file_path):
					im = imageio.imread(file_path)
					if self._sample_shape is None:
						self._sample_shape = im.shape
					else:
						assert(self._sample_shape == im.shape)
					self._data['x'].append(im)
					cls_onehot = np.zeros(self._num_classes)
					cls_onehot[int(cls)] = 1
					self._data['y'].append(cls_onehot)

		assert(len(self._data['x']) == len(self._data['y']))

	def _load_from_h5(self, path):
		with h5py.File(path, 'r') as f:
			keys = list(f.keys())

			if 'x' in keys and 'y' in keys:
				x = np.array(f['x'].value)
				y = np.array(f['y'].value)
			else:
				raise Exception('Data not found')


		if x.shape[1] < x.shape[-1]:
			x = np.moveaxis(x, 1, -1)
		self._sample_shape = x.shape[1:]
		self._num_classes = np.unique(y).size

		y_onehot = np.zeros((y.size, self.num_classes))
		y_onehot[np.arange(y.size), y] = 1

		self._data = {}
		self._data['x'] = list(x)
		self._data['y'] = list(y_onehot)


	def _load_cifar10(self, split):
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

		self._data = {}
		self._data['x'] = []
		self._data['y'] = []
		self._sample_shape = (32,32,3)

		if split == 'train':
			self._data['x'] = x_train
			self._data['y'] = y_train
		elif split == 'val' or split == 'test':
			xt, xv, yt, yv = train_test_split(x_test, y_test, test_size=0.2, random_state=1)

			if split == 'val':
				self._data['x'] = xv
				self._data['y'] = yv
			elif split == 'test':
				self._data['x'] = xt
				self._data['y'] = yt


		y = np.array(self._data['y']).ravel()
		y_onehot = np.zeros((y.size, 10))
		y_onehot[np.arange(y.size), y] = 1
		self._data['y'] = list(y_onehot)
		self.num_classes = 10

		# Upscale
		self._resize_data(32, 32)

	def _load_cinic10(self, split):
		images = []
		y_names = []

		train_path = "../datasets/CINIC/train/"
		valid_path = "../datasets/CINIC/valid/"
		test_path = "../datasets/CINIC/test/"

		if split == 'train':
			path = train_path
		elif split == 'val':
			path = valid_path
		elif split == 'test':
			path = test_path
		else:
			print('Not valid split')
			return

		for label_path in os.listdir(path):
			label = label_path
			for image_path in os.listdir(train_path + "/" + label_path):
				img = io.imread(train_path + "/" + label_path + "/" + image_path)
				if img.shape != (32, 32, 3):
					continue
				images.append(img)
				y_names.append(label)
		images = np.concatenate([arr[np.newaxis] for arr in images])

		categories = pd.Categorical(y_names)
		y = categories.codes
		x = images

		self.num_classes = 10
		self._sample_shape = x.shape[1:]
		self._data['x'] = x.astype('float32')
		self._data['y'] = list(keras.utils.to_categorical(y, self.num_classes))


	def _load_mnist(self, split):
		(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

		self._data = {}
		self._data['x'] = []
		self._data['y'] = []
		self._sample_shape = (32,32, 1)

		if split == 'train':
			self._data['x'] = x_train
			self._data['y'] = y_train
		elif split == 'val' or split == 'test':
			xt, xv, yt, yv = train_test_split(x_test, y_test, test_size=0.2, random_state=1)

			if split == 'val':
				self._data['x'] = xv
				self._data['y'] = yv
			elif split == 'test':
				self._data['x'] = xt
				self._data['y'] = yt


		y = np.array(self._data['y']).ravel()
		self.num_classes = 10
		self._data['y'] = list(keras.utils.to_categorical(y, self.num_classes))
		self._data['x'] = np.expand_dims(self._data['x'], axis=-1)

		# Upscale
		self._resize_data(32, 32)


	def _load_wiki(self, split):
		train_path = '../datasets/wiki_crop/data_processed/train.csv'
		val_path = '../datasets/wiki_crop/data_processed/val.csv'
		test_path = '../datasets/wiki_crop/data_processed/test.csv'
		images_path = '../datasets/wiki_crop/data_processed/'

		path = val_path if split == 'val' else test_path if split == 'test' else train_path

		self._sample_shape = (128, 128, 3)
		self.num_classes = 8

		df = pd.read_csv(path)

		self._data['x'] = []
		self._data['y'] = list(df['age_cat'])

		for path in df['path']:
			img = io.imread(os.path.join(images_path, path))

			if len(img.shape) < 3:
				img = np.stack((img,)*3, axis=-1)

			self._data['x'].append(img)

		self._data['x'] = np.concatenate([arr[np.newaxis] for arr in self._data['x']])
		self._data['y'] = keras.utils.to_categorical(self._data['y'], self.num_classes)


	def _load_imdb(self, split):
		train_path = '../datasets/imdb_crop/data_processed/train.csv'
		val_path = '../datasets/imdb_crop/data_processed/val.csv'
		test_path = '../datasets/imdb_crop/data_processed/test.csv'
		images_path = '../datasets/imdb_crop/data_processed/'

		path = val_path if split == 'val' else test_path if split == 'test' else train_path

		self._sample_shape = (128, 128, 3)
		self.num_classes = 8

		df = pd.read_csv(path)

		self._data['x'] = []
		self._data['y'] = list(df['age_cat'])

		for path in df['path']:
			img = io.imread(os.path.join(images_path, path))

			if len(img.shape) < 3:
				img = np.stack((img,)*3, axis=-1)

			self._data['x'].append(img)

		self._data['x'] = np.concatenate([arr[np.newaxis] for arr in self._data['x']])
		self._data['y'] = keras.utils.to_categorical(self._data['y'], self.num_classes)



	def _resize_data(self, width, height):
		data_resized = np.zeros((self._data['x'].shape[0], width, height, self.num_channels))
		for i, img in enumerate(self._data['x']):
			img_resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
			# cv2 returns 2 dims array when using non rgb images but we need 3 dims
			if len(img_resized.shape) < 3:
				img_resized = np.expand_dims(img_resized, axis=-1)
			data_resized[i] = img_resized

		self._data['x'] = data_resized

	def standardize_data(self, mean=None, std=None):
		x = np.array(self._data['x'])
		if mean is None:
			mean = x.mean()

		if std is None:
			std = x.std()

		self._data['x'] = list((x - mean) / std)

		return mean, std


	def size(self):
		"""
		Get dataset size.
		:return: number of samples.
		"""
		return len(self._data['y'])

	def num_batches(self, batch_size):
		"""
		Get number of batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size() / batch_size)

	def get_class_weights(self):
		"""
		Get class weights that you can use to counter-act the dataset unbalance.
		Class weights are calculated based on the frequency of each class.
		:return: dictionary that contains the weight for each class.
		"""

		# No weights if no data
		if not self._data or not self._data['y']:
			return {}

		y_label = np.argmax(self._data['y'], axis=1)
		return compute_class_weight('balanced', np.unique(y_label), y_label)

	@property
	def num_channels(self):
		"""
		Get number of channels of the images.
		:return: number of channels.
		"""
		return len(self.sample_shape) == 3 and self.sample_shape[2] or 1

	@property
	def img_size(self):
		"""
		Get image size for squared images.
		:return: image size (integer).
		"""
		return self.sample_shape[0]

	def is_rgb(self):
		"""
		Check whether the images are RGB.
		:return:
		"""
		return self.num_channels == 3