import os
import pandas as pd
import numpy as np
from shutil import copyfile
from sklearn.model_selection import StratifiedShuffleSplit

train_path = '../retinopathy/train_small_bmp'
train_labels = '../retinopathy/trainLabels.csv'
train_ext = '.bmp'
test_path = '../retinopathy/test_small_bmp'
test_labels = '../retinopathy/testLabels.csv'
test_ext = '.bmp'

train_dest = '../retinopathy/256/train'
val_dest = '../retinopathy/256/val'
test_dest = '../retinopathy/256/test'

try:
	os.makedirs(train_dest)
except OSError as exc:
	pass

try:
	os.makedirs(val_dest)
except OSError as exc:
	pass

try:
	os.makedirs(test_dest)
except OSError as exc:
	pass

# TRAIN

df = pd.read_csv(train_labels)
data = np.array(df)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
splits = sss.split(data[:,0], data[:,1])
for train_indices, val_indices in splits:
	for train_index in train_indices:
		img, label = data[train_index]
		try:
			os.makedirs(os.path.join(train_dest, str(label)))
		except OSError as exc:
			pass

		src = os.path.join(train_path, "{}{}".format(img, train_ext))
		dst = os.path.join(train_dest, str(label), "{}{}".format(img, train_ext))

		if os.path.isfile(src) and not os.path.isfile(dst):
			copyfile(src, dst)
			print("Copied {} to {}".format(src, dst))

	for val_index in val_indices:
		img, label = data[val_index]
		try:
			os.makedirs(os.path.join(val_dest, str(label)))
		except OSError as exc:
			pass

		src = os.path.join(train_path, "{}{}".format(img, train_ext))
		dst = os.path.join(val_dest, str(label), "{}{}".format(img, train_ext))

		if os.path.isfile(src) and not os.path.isfile(dst):
			copyfile(src, dst)
			print("Copied {} to {}".format(src, dst))

# TEST

df = pd.read_csv(test_labels)
data = np.array(df)

for img, label, _ in data:
	try:
		os.makedirs(os.path.join(test_dest, str(label)))
	except OSError as exc:
		pass

	src = os.path.join(test_path, "{}{}".format(img, test_ext))
	dst = os.path.join(test_dest, str(label), "{}{}".format(img, test_ext))

	if os.path.isfile(src) and not os.path.isfile(dst):
		copyfile(src, dst)
		print("Copied {} to {}".format(src, dst))