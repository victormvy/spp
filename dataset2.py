import imageio
import numpy as np
import os
import math
import h5py
from sklearn.model_selection import train_test_split
import keras
import cv2
import pandas as pd
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from generators import SmallGenerator, BigGenerator


class Dataset:
	"""
	Class that represents a dataset that is loaded from a file.
	"""
	def __init__(self, name):
		# Name / path of the dataset
		self._name = name

		# Load status
		self._loaded = False
		self._big_dataset = False

		# Numpy arrays for small datasets
		self._x_train = None
		self._y_train = None
		self._x_val = None
		self._y_val = None
		self._x_test = None
		self._y_test = None

		# Dataframes for big datasets
		self._df_train = None
		self._df_val = None
		self._df_test = None

		# Set dataframes x and y columns
		self._x_col = 'path'
		self._y_col = 'y'

		# Base path for images of big datasets
		self._base_path = None

		# Generator for each dataset split
		self._train_generator = None
		self._val_generator = None
		self._test_generator = None

		# Store means and std to avoid multiple calculations
		self._mean_train = None
		self._mean_val = None
		self._mean_test = None
		self._std_train = None
		self._std_val = None
		self._std_test = None

		super(Dataset, self).__init__()

	def load(self, name):
		if hasattr(self, "_load_" + name):
			return getattr(self, "_load_" + name)()
		else:
			raise Exception('Invalid dataset.')

	def _load_cifar10(self):
		# Small dataset
		self._big_dataset = False

		# Set sample shape and number of classes
		self._sample_shape = (32, 32, 3)
		self.num_classes = 10

		# Load data
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
		x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=1)

		# Save x and y
		self._x_train, self._y_train = x_train, y_train
		self._x_val, self._y_val = x_val, y_val
		self._x_test, self._y_test = x_test, y_test

		# Mark dataset as loaded
		self._loaded = True

	def _load_mnist(self):
		# Small dataset
		self._big_dataset = False

		# Set sample shape and number of classes
		self._sample_shape = (32, 32, 1)
		self.num_classes = 10

		# Load data
		(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
		x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2, random_state=1)

		# Upscale
		x_train = self._resize_data(x_train, 32, 32, self.num_channels)
		x_val = self._resize_data(x_val, 32, 32, self.num_channels)
		x_test = self._resize_data(x_test, 32, 32, self.num_channels)

		# Save x and y
		self._x_train, self._y_train = x_train, y_train
		self._x_val, self._y_val = x_val, y_val
		self._x_test, self._y_test = x_test, y_test

		# Mark dataset as loaded
		self._loaded = True

	def _load_wiki(self):
		# Small dataset
		self._big_dataset = False

		# Load dataframes
		df_train = pd.read_csv('../datasets/wiki_crop/data_processed/train.csv')
		df_val = pd.read_csv('../datasets/wiki_crop/data_processed/val.csv')
		df_test = pd.read_csv('../datasets/wiki_crop/data_processed/test.csv')
		
		# Base path for images
		base_path = '../datasets/wiki_crop/data_processed/'

		# Dataframe columns
		x_col = 'path'
		y_col = 'age_cat'

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self.num_classes = 8

		# Load data from dataframe
		self._x_train, self._y_train = self._load_from_dataframe(df_train, x_col, y_col, base_path)
		self._x_val, self._y_val = self._load_from_dataframe(df_val, x_col, y_col, base_path)
		self._x_test, self._y_test = self._load_from_dataframe(df_test, x_col, y_col, base_path)

		# Mark dataset as loaded
		self._loaded = True

	def _load_imdb(self):
		# Big dataset
		self._big_dataset = True

		# Load dataframes
		self._df_train = pd.read_csv('../datasets/imdb_crop/data_processed/train.csv')
		self._df_val = pd.read_csv('../datasets/imdb_crop/data_processed/val.csv')
		self._df_test = pd.read_csv('../datasets/imdb_crop/data_processed/test.csv')

		# Set x and y columns
		self._x_col = 'path'
		self._y_col = 'age_cat'

		# Set base path for images
		self._base_path = '../datasets/imdb/data_processed/'

		# Set sample shape and number of classes
		self._sample_shape = (128, 128, 3)
		self._num_classes = 8

		# Check that images exist
		if self._check_dataframe_images(self._df_train, self._x_col, self._base_path) and \
        self._check_dataframe_images(self._df_val, self._x_col, self._base_path) and \
        self._check_dataframe_images(self._df_test, self._x_col, self._base_path):
			# If everything is correct, mark dataset as loaded
			self._loaded = True

	# Fully load x and y from dataframe
	def _load_from_dataframe(self, df, x_col, y_col, base_path):
		x = []
		y = list(df[y_col])

		for path in df['path']:
			img = imread(os.path.join(base_path, path))

			if len(img.shape) < 3:
				img = np.stack((img,)*3, axis=-1)

			x.append(img)
		
		x = np.concatenate([arr[np.newaxis] for arr in x])

		return x, y

	# Resize array of images
	def _resize_data(self, x, width, height, channels):
		x_resized = np.zeros((x.shape[0], width, height, channels))
		for i, img in enumerate(x):
			img_resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
			# cv2 returns 2 dims array when using non rgb images but we need 3 dims
			if len(img_resized.shape) < 3:
				img_resized = np.expand_dims(img_resized, axis=-1)
			x_resized[i] = img_resized

		return x_resized

	def generate_train(self, batch_size, augmentation):
		if self._big_dataset:
			return BigGenerator(self._df_train, self._base_path, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size, augmentation=augmentation)
		else:
			return SmallGenerator(self._x_train, self._y_train, mean=self.mean_train, std=self.std_train, batch_size=batch_size, augmentation=augmentation)

	def generate_val(self, batch_size):
		if self._big_dataset:
			return BigGenerator(self._df_val, self._base_path, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size)
		else:
			return SmallGenerator(self._x_val, self._y_val, mean=self.mean_train, std=self.std_train, batch_size=batch_size)

	def generate_test(self, batch_size):
		if self._big_dataset:
			return BigGenerator(self._df_test, self._base_path, self._x_col, self._y_col, mean=self.mean_train, std=self.std_train, batch_size=batch_size)
		else:
			return SmallGenerator(self._x_test, self._y_test, mean=self.mean_train, std=self.std_train, batch_size=batch_size)


	def _check_dataframe_images(self, df, x_col, base_path):
		for path in df[x_col]:
			if not os.path.exists(os.path.join(base_path, path)):
				return False
		return True

	def _mean_small(self, x):
		return x.mean()

	def _mean_big(self, df):
		paths = df[self._x_col].values
		count = df.shape[0]

		mean = 0
		for path in paths:
			img = io.imread(os.path.join(self._base_path, path))
			mean += img.mean()
		
		mean /= count

		return mean

	def _std_small(self, x):
		return x.std()

	def _std_big(self, df):
		paths = df[self._x_col].values

		n = 0
		summ = 0
		sumsq = 0

		for path in paths:
			img = io.imread(os.path.join(self._base_path, path))
			n += np.array(img.shape).prod()
			summ += img.sum()
			sumsq += pow(img.sum(), 2)

		var = (sumsq - pow(sum, 2) / n) / (n - 1)

		return math.sqrt(var)

	@property
	def mean_train(self):
		if not self._mean_train:
			self._mean_train = self._mean_big(self._df_train) if self._big_dataset else self._mean_small(self._x_train)
		return self._mean_train

	@property
	def mean_val(self):
		if not self._mean_val:
			self._mean_val = self._mean_big(self._df_val) if self._big_dataset else self._mean_small(self._x_val)
		return self._mean_val

	@property
	def mean_test(self):
		if not self._mean_test:
			self._mean_test = self._mean_big(self._df_test) if self._big_dataset else self._mean_small(self._x_test)
		return self._mean_test

	@property
	def std_train(self):
		if not self._std_train:
			self._std_train = self._std_big(self._df_train) if self._big_dataset else self._std_small(self._x_train)
		return self._std_train

	@property
	def std_val(self):
		if not self._std_val:
			self._std_val = self._std_big(self._df_val) if self._big_dataset else self._std_small(self._x_val)
		return self._std_val

	@property
	def std_test(self):
		if not self._std_test:
			self._std_test = self._std_big(self._df_test) if self._big_dataset else self._std_small(self._x_test)
		return self._std_test

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

	def size_train(self):
		"""
		Get dataset train size.
		:return: number of samples.
		"""
		return 0 if not self._loaded else self._df_train.shape[0] if self._big_dataset else self._y_train.shape[0]

	def size_val(self):
		"""
		Get dataset val size.
		:return: number of samples.
		"""
		return 0 if not self._loaded else self._df_val.shape[0] if self._big_dataset else self._y_val.shape[0]

	def size_test(self):
		"""
		Get dataset test size.
		:return: number of samples.
		"""
		return 0 if not self._loaded else self._df_test.shape[0] if self._big_dataset else self._y_test.shape[0]

	def num_batches_train(self, batch_size):
		"""
		Get number of train batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_train() / batch_size)

	def num_batches_val(self, batch_size):
		"""
		Get number of val batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_val() / batch_size)

	def num_batches_test(self, batch_size):
		"""
		Get number of test batches for a given batch size.
		:param batch_size: batch size.
		:return: number of batches.
		"""
		return math.ceil(self.size_test() / batch_size)

	def get_class_weights(self):
		"""
		Get class weights that you can use to counter-act the dataset unbalance.
		Class weights are calculated based on the frequency of each class.
		:return: dictionary that contains the weight for each class.
		"""

		# No weights if not loaded
		if not self._loaded:
			return {}

		y_label = self._df_train[self._y_col] if self._big_dataset else self._y_train

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