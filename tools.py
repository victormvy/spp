# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import os
import prettytable
import h5py
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

evaluation_file = 'evaluation.pickle'


def resume_one_metric(metric, results_path):
	t = prettytable.PrettyTable(['Dataset', 'BS', 'LR', 'LF', 'ACT', 'Train ' + metric, 'Mean Tr', 'Validation ' + metric, 'Mean V', 'Test ' + metric, 'Mean Te'])
	for item in sorted(os.listdir(results_path)):
		if os.path.isdir(os.path.join(results_path, item)):
			train, val, test = '', '', ''
			train_values, val_values, test_values = np.array([]), np.array([]), np.array([])
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)

						if metric in p['metrics']['Train'] and metric in p['metrics']['Validation'] and metric in p['metrics']['Test']:
							train += '{:.5} '.format(round(p['metrics']['Train'][metric], 5))
							val += '{:.5} '.format(round(p['metrics']['Validation'][metric], 5))
							test += '{:.5} '.format(round(p['metrics']['Test'][metric], 5))

							# Accumulate sums
							train_values = np.append(train_values, p['metrics']['Train'][metric])
							val_values = np.append(val_values, p['metrics']['Validation'][metric])
							test_values = np.append(test_values, p['metrics']['Test'][metric])

			if 'p' in locals():
				t.add_row([
					p['config']['db'],
					p['config']['batch_size'],
					p['config']['lr'],
					p['config']['final_activation'],
					p['config']['activation'],
					train,
					train_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(train_values), 5), round(np.std(train_values, ddof=min(1, len(train_values)-1)), 5)) or 0,
					val,
					val_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(val_values), 5), round(np.std(val_values, ddof=min(1, len(val_values)-1)), 5)) or 0,
					test,
					test_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(test_values), 5), round(np.std(test_values, ddof=min(1, len(test_values)-1)), 5)) or 0
				])


	print(t)


def show_confusion_matrices(results_path):
	t = prettytable.PrettyTable(
		['Dataset', 'BS', 'LR', 'LF', 'ACT', 'Execution', 'Train CF', 'Validation CF', 'Test CF'], hrules=prettytable.ALL)
	for item in sorted(os.listdir(results_path)):
		if os.path.isdir(os.path.join(results_path, item)):
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(
						os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)

						if 'Confusion matrix' in p['metrics']['Train'] and 'Confusion matrix' \
								in p['metrics']['Validation'] and 'Confusion matrix' in	p['metrics']['Test']:

							t.add_row([
								p['config']['db'],
								p['config']['batch_size'],
								p['config']['lr'],
								p['config']['final_activation'],
								p['config']['activation'],
								item2,
								p['metrics']['Train']['Confusion matrix'],
								p['metrics']['Validation']['Confusion matrix'],
								p['metrics']['Test']['Confusion matrix']
							])

	print(t)


def show_latex_table(results_path, show_std=False):
	header, train, val, test = '', '', '', ''
	for item in sorted(os.listdir(results_path)):
		if os.path.isdir(os.path.join(results_path, item)):
			metrics = {'Train' : {}, 'Validation' : {}, 'Test' : {}}
			count = len(os.listdir(os.path.join(results_path, item)))
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(
						os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p_aux = pickle.load(f)
						if p_aux is not None:
							p = p_aux

						for metric, value in p['metrics']['Train'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Train']:
									metrics['Train'][metric].append(value)
								else:
									metrics['Train'][metric] = [value]

						for metric, value in p['metrics']['Validation'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Validation']:
									metrics['Validation'][metric].append(value)
								else:
									metrics['Validation'][metric] = [value]

						for metric, value in p['metrics']['Test'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Test']:
									metrics['Test'][metric].append(value)
								else:
									metrics['Test'][metric] = [value]

			if not 'p' in locals():
				continue

			if header == '':
				header = 'Dataset & BS & LF & LR & ACT'

				for metric, value in metrics['Train'].items():
					if metric != 'Confusion matrix':
						header += ' & $\overline{{\\text{{{}}}}}{}$'.format(metric, '_{{(SD)}}' if show_std else '')

				header += '\\\\\\hline'

			t = '{} & {} & {} & {}'.format(
				p['config']['db'],
				p['config']['batch_size'],
				p['config']['final_activation'].replace('poml', 'logit')
												 .replace('pomp', 'probit')
												 .replace('pomclog', 'c log-log'),
				p['config']['activation'],
			'${:.0E}}}$'.format(p['config']['lr']).replace('E-0', '0^{-')
				.replace('E+0', '0^{+')
			)

			train += t
			val += t
			test += t

			for metric, values in metrics['Train'].items():
				if metric != 'Confusion matrix':
					train += ' & ${:.3f}'.format(round(np.mean(values), 3))
					if show_std:
						train += '_{{({:.3f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 3))
					train += '$'

			train += '\\\\\n'

			for metric, values in metrics['Validation'].items():
				if metric != 'Confusion matrix':
					val += ' & ${:.3f}'.format(round(np.mean(values), 3))
					if show_std:
						val += '_{{({:.3f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 3))
					val += '$'

			val += '\\\\\n'

			for metric, values in metrics['Test'].items():
				if metric != 'Confusion matrix':
					test += ' & ${:.3f}'.format(round(np.mean(values), 3))
					if show_std:
						test += '_{{({:.3f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 3))
					test += '$'

			test += '\\\\\n'

	print('===== TRAIN =====')
	print(header)
	print(train)

	print('===== VALIDATION =====')
	print(header)
	print(val)

	print('===== TEST =====')
	print(header)
	print(test)

def create_h5_dataset(path, file):
	x = []
	y = []
	for dir in os.listdir(path):
		cls = int(dir)
		for f in os.listdir(os.path.join(path, dir)):
			full_f = os.path.join(path, dir, f)
			if os.path.isfile(full_f):
				data = np.array(Image.open(full_f))
				x.append(data)
				y.append(cls)

	x = np.array(x) / float(np.max(x))
	y = np.array(y)
	print(x.shape)
	print(y.shape)

	# Standardize each color channel
	means = x.mean(axis=(0,1,2))
	stds = x.std(axis=(0,1,2), ddof=1)
	x = (x - means) / stds

	f = h5py.File(file, 'w')
	f.create_dataset('x', data = x, compression = 'gzip', compression_opts = 9)
	f.create_dataset('y', data = y, compression = 'gzip', compression_opts = 9)

def create_h5_cifar10(file, shape):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	train_rs_op = tf.image.resize_images(x_train, shape, method=tf.image.ResizeMethod.BILINEAR)
	test_rs_op = tf.image.resize_images(x_test, shape, method=tf.image.ResizeMethod.BILINEAR)
	val_perc = 0.2

	with tf.Session() as sess:
		print('Resizing images to {}'.format(shape))
		train_rs, test_rs = sess.run([train_rs_op, test_rs_op])

	print('Splitting {} for validation'.format(val_perc))
	train_x, val_x, train_y, val_y = train_test_split(train_rs, y_train, test_size=val_perc)
	test_x, test_y = test_rs, y_test

	train_file = '{}_train.h5'.format(file)
	val_file = '{}_val.h5'.format(file)
	test_file = '{}_test.h5'.format(file)

	print('Saving train set to {}...'.format(train_file))
	with h5py.File(train_file, 'w') as f:
		f.create_dataset('x', data=train_x, compression='gzip', compression_opts=9)
		f.create_dataset('y', data=np.ravel(train_y), compression='gzip', compression_opts=9)

	print('Saving validation set to {}...'.format(val_file))
	with h5py.File(val_file, 'w') as f:
		f.create_dataset('x', data=val_x, compression='gzip', compression_opts=9)
		f.create_dataset('y', data=np.ravel(val_y), compression='gzip', compression_opts=9)

	print('Saving test set to {}...'.format(test_file))
	with h5py.File(test_file, 'w') as f:
		f.create_dataset('x', data=test_x, compression='gzip', compression_opts=9)
		f.create_dataset('y', data=np.ravel(test_y), compression='gzip', compression_opts=9)

def display_h5_images(file):
	with h5py.File(file, 'r') as f:
		if not 'x' in f:
			return Exception('Data not found')
		x = f['x'].value
		for item in x:
			img = Image.fromarray(item.astype(np.uint8), 'RGB')
			img.show()
			inp = input('Press a key to continue or write q to exit.')
			if inp == 'q':
				return


def option_resume_one_metric():
	results_path = input('Results path: ')
	metric = input('Metric name: ')
	resume_one_metric(metric, results_path)

def option_show_confusion_matrices():
	results_path = input('Results path: ')
	show_confusion_matrices(results_path)

def option_latex_table():
	results_path = input('Results path: ')

	print('=====================')
	print('1. Mean')
	print('2. Mean and std')
	print('=====================')
	option = input(' Choose one option: ')

	show_latex_table(results_path, option == '2')


def option_create_h5_dataset():
	path = input('Path: ')
	file = input('Output file: ')
	create_h5_dataset(path, file)

def option_create_h5_cifar10():
	name = input('Output name (output file: <name>_[train/val/test].h5): ')
	sz = int(input('Image size: '))
	create_h5_cifar10(name, (sz,sz))

def option_display_h5_images():
	file = input('H5 file: ')
	display_h5_images(file)

def show_menu():
	print('=====================================')
	print('1. Resume results for one metric')
	print('2. Show confusion matrices')
	print('3. Show latex table')
	print('4. Create h5 dataset')
	print('5. Create h5 cifar10')
	print('6. Display images from h5')
	print('=====================================')
	option = input(' Choose one option: ')

	return option


def select_option(option):
	if option == '1':
		option_resume_one_metric()
	elif option == '2':
		option_show_confusion_matrices()
	elif option == '3':
		option_latex_table()
	elif option == '4':
		option_create_h5_dataset()
	elif option == '5':
		option_create_h5_cifar10()
	elif option == '6':
		option_display_h5_images()


if __name__ == '__main__':
	select_option(show_menu())
