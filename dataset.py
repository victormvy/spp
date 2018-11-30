import imageio
import numpy as np
import os
import math
import h5py

class Dataset():
	"""
	Class that represents a dataset that is loaded from a file.
	"""
	def __init__(self, path=""):
		self._data = {'x' : [], 'y' : []}
		self._num_classes = 0
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

	def load(self, path):
		if os.path.isdir(path):
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
			for f in os.listdir(os.path.join(path, cls)):
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
		pass

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

		counts = {}
		total_count = 0
		for lbl_oh in self._data['y']:
			label = np.argmax(lbl_oh)
			total_count += 1
			if label in counts:
				counts[label] += 1
			else:
				counts[label] = 1

		weights = {}

		for k in counts:
			weights[k] = np.log(total_count / counts[k])

		return weights

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