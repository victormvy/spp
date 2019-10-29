import math
import numpy as np
from skimage.io import imread


def std():
	paths = ['../datasets/imdb_crop/data_processed/00/nm0000100_rm12818688_1955-1-6_2003.jpg',
			 '../datasets/imdb_crop/data_processed/00/nm0000100_rm46373120_1955-1-6_2003.jpg']
	n = 0
	summ = 0
	sumsq = 0

	for path in paths:
		img = imread(path)
		n += np.array(img.shape).prod()
		summ += img.sum()
		sumsq += pow(img, 2).sum()

	var = (sumsq / n - pow(summ / n, 2))

	print(var)

	return math.sqrt(var)

def std2():
	paths = ['../datasets/imdb_crop/data_processed/00/nm0000100_rm12818688_1955-1-6_2003.jpg',
			 '../datasets/imdb_crop/data_processed/00/nm0000100_rm46373120_1955-1-6_2003.jpg']

	imgs = np.array([imread(paths[0]), imread(paths[1])])

	return imgs.std()


def online_variance():
	paths = ['../datasets/imdb_crop/data_processed/00/nm0000100_rm12818688_1955-1-6_2003.jpg',
			 '../datasets/imdb_crop/data_processed/00/nm0000100_rm46373120_1955-1-6_2003.jpg']

	n = 0
	mean = 0
	M2 = 0

	data = []
	for path in paths:
		data.append(imread(path))
	data = np.array(data).ravel()

	for x in data:
		n += 1
		delta = x - mean
		mean = mean + delta/n
		M2 = M2 + delta*(x - mean)

	variance = M2/(n - 1)
	return variance


print(std2())


def std3():
	paths = ['../datasets/imdb_crop/data_processed/00/nm0000100_rm12818688_1955-1-6_2003.jpg',
			 '../datasets/imdb_crop/data_processed/00/nm0000100_rm46373120_1955-1-6_2003.jpg']

	imgs = np.array([imread(paths[0]), imread(paths[1])])

	mean = imgs.mean()

	sums = 0

	for img in imgs:
		sums += ((img - mean) ** 2) / len(imgs)

	print(sums.shape)

	std = np.sqrt(np.mean(sums))

	return std

print(std3())

def std4():
	paths = ['../datasets/imdb_crop/data_processed/00/nm0000100_rm12818688_1955-1-6_2003.jpg',
			 '../datasets/imdb_crop/data_processed/00/nm0000100_rm46373120_1955-1-6_2003.jpg']

	imgs = np.array([imread(paths[0]), imread(paths[1])])

	mean = imgs.mean()

	sums = 0

	for img in imgs:
		sums += np.sum(((img - mean) ** 2) / len(imgs))

	std = np.sqrt(sums / np.prod(imgs[0].shape))

	return std

print(std4())