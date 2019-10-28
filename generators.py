from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool
from skimage.io import imread
import numpy as np
import os

class SmallGenerator(Sequence):
    def __init__(self, x, y, mean=None, std=None, batch_size=128, augmentation={}, workers=7, one_hot=True):
        self._x = x
        self._y = y
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._one_hot = one_hot

    def __len__(self):
        return int(np.ceil(len(self._x) / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_x = self._x[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_y = self._y[idx * self._batch_size:(idx + 1) * self._batch_size]

        batch_x = self._p.map(self._process_data, batch_x)

        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._one_hot:
            batch_y = to_categorical(batch_y, num_classes=np.unique(batch_y))

        return np.array(batch_x), np.array(batch_y)

    # Process a single image
    def _process_data(self, x):
        # Apply data augmentation
        if len(self._augmentation > 0):
            x = ImageDataGenerator.apply_transform(x, self._augmentation)
        return x


class BigGenerator(Sequence):
    def __init__(self, df, base_path, x_col='x', y_col='y', mean=None, std=None, batch_size=128, augmentation={}, workers=7, one_hot=True, force_rgb=True):
        self._df = df
        self._base_path = base_path
        self._x_col = x_col
        self._y_col = y_col
        self._mean = mean
        self._std = std
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._workers = workers
        self._p = Pool(self._workers)
        self._one_hot = one_hot
        self._force_rgb = force_rgb

    def __len__(self):
        return int(np.ceil(self._df.shape[0] / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_paths = self._df.loc[idx * self._batch_size : (idx + 1) * self._batch_size, self._x_col]
        batch_y = self._df.loc[idx * self._batch_size : (idx + 1) * self._batch_size, self._y_col]

        # Load batch images using multiprocessing
        batch_x = self._p.map(self._process_data, batch_paths)

        # Standardize
        if self._mean and self._std:
            batch_x = (batch_x - self._mean) / self._std

        if self._one_hot:
            batch_y = to_categorical(batch_y, num_classes=np.unique(batch_y))

        return np.array(batch_x), np.array(batch_y)

    def _process_data(self, path):
        img = imread(os.path.join(self._base_path, path))

        # Convert to RGB if grayscale
        if self._force_rgb and len(img.shape) < 3:
            img = np.stack((img,)*3, axis=-1)

        # Apply data augmentation
        if len(self._augmentation > 0):
            img = ImageDataGenerator.apply_transform(img, self._augmentation)

        return img