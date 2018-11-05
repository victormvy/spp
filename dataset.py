import imageio
import numpy as np
import os
import math

class Dataset():

	def __init__(self, path=""):
		self._data = {'x' : [], 'y' : []}
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
		self._data.data = data

	@data.deleter
	def data(self):
		del self._data

	def load(self, path):
		self._data = {}
		self._data['x'] = []
		self._data['y'] = []
		num_classes = len(os.listdir(path))
		for cls in os.listdir(path):
			for f in os.listdir(os.path.join(path, cls)):
				file_path = os.path.join(os.path.join(path, cls), f)
				if os.path.isfile(file_path):
					im = imageio.imread(file_path)
					self._data['x'].append(im)
					cls_onehot = np.zeros(num_classes)
					cls_onehot[int(cls)] = 1
					self._data['y'].append(cls_onehot)

		assert(len(self._data['x']) == len(self._data['y']))

	def size(self):
		return len(self._data['y'])

	def num_batches(self, batch_size):
		return math.ceil(self.size() / batch_size)


