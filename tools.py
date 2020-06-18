# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import os
import prettytable
import h5py
import keras
import cv2
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from shutil import copy
from keras import backend as K

evaluation_file = 'evaluation.pickle'


def resume_one_metric(metric, results_path):
	t = prettytable.PrettyTable(['#','Dataset', 'BS', 'LR', 'LF', 'ACT', 'Train ' + metric, 'Mean Tr', 'Validation ' + metric, 'Mean V', 'Test ' + metric, 'Mean Te'])
	t2 = prettytable.PrettyTable(['#', 'Name'])
	for i, item in enumerate(sorted(os.listdir(results_path))):
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
					i,
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

				t2.add_row([i, item])

	print(t2)
	print(t)


def resume_one_metric_csv(metric, results_path, csv_path):
	t = []
	max_train_values = 0
	max_val_values = 0
	max_test_values = 0

	for i, item in enumerate(sorted(os.listdir(results_path))):
		if os.path.isdir(os.path.join(results_path, item)):
			train_values, val_values, test_values = np.array([]), np.array([]), np.array([])
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)

						if metric in p['metrics']['Train'] and metric in p['metrics']['Validation'] and metric in p['metrics']['Test']:
							# Accumulate sums
							train_values = np.append(train_values, p['metrics']['Train'][metric])
							val_values = np.append(val_values, p['metrics']['Validation'][metric])
							test_values = np.append(test_values, p['metrics']['Test'][metric])

			max_train_values = train_values.size if train_values.size > max_train_values else max_train_values
			max_val_values = val_values.size if val_values.size > max_val_values else max_val_values
			max_test_values = test_values.size if test_values.size > max_test_values else max_test_values


	for i, item in enumerate(sorted(os.listdir(results_path))):
		if os.path.isdir(os.path.join(results_path, item)):
			train_values, val_values, test_values = np.array([]), np.array([]), np.array([])
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)

						if metric in p['metrics']['Train'] and metric in p['metrics']['Validation'] and metric in p['metrics']['Test']:
							# Accumulate sums
							train_values = np.append(train_values, p['metrics']['Train'][metric])
							val_values = np.append(val_values, p['metrics']['Validation'][metric])
							test_values = np.append(test_values, p['metrics']['Test'][metric])

			if 'p' in locals():
				train_mean = train_values.size > 0 and round(np.mean(train_values), 5) or 0
				train_std = train_values.size > 0 and round(np.std(train_values), 5) or 0
				val_mean = val_values.size > 0 and round(np.mean(val_values), 5) or 0
				val_std = val_values.size > 0 and round(np.std(val_values), 5) or 0
				test_mean = test_values.size > 0 and round(np.mean(test_values), 5) or 0
				test_std = test_values.size > 0 and round(np.std(test_values), 5) or 0
				train_values.resize(max_train_values, refcheck=False)
				val_values.resize(max_val_values, refcheck=False)
				test_values.resize(max_test_values, refcheck=False)

				t.append([
					i,
					item,
					p['config']['db'],
					p['config']['batch_size'],
					p['config']['lr'],
					p['config']['final_activation'],
					p['config']['activation'],
					*list(train_values.round(5)),
					train_mean,
					train_std,
					*list(val_values.round(5)),
					val_mean,
					val_std,
					*list(test_values.round(5)),
					test_mean,
					test_std
				])

	header = ['#', 'Name', 'Dataset', 'BS', 'LR', 'LF', 'ACT']
	
	for i in range(max_train_values):
		header.append('Train ' + metric + str(i))

	header.append('Train mean')
	header.append('Train std')

	for i in range(max_val_values):
		header.append('Val ' + metric + str(i))

	header.append('Val mean')
	header.append('Val std')

	for i in range(max_test_values):
		header.append('Test ' + metric + str(i))

	header.append('Test mean')
	header.append('Test std')

	t = [header] + t

	with open(csv_path, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=';')
		writer.writerows(t)


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
							if metric != 'Confusion matrix' and metric != 'OMAE':
								if metric in metrics['Test']:
									metrics['Test'][metric].append(value)
								else:
									metrics['Test'][metric] = [value]

			if not 'p' in locals():
				continue

			if header == '':
				header = 'Dataset & BS & LF & LR & ACT'

				for metric, value in metrics['Train'].items():
					if metric != 'Confusion matrix' and metric != 'OMAE':
						header += ' & $\overline{{\\text{{{}}}}}{}$'.format(metric, '_{{(SD)}}' if show_std else '')

				header += '\\\\\\hline'

			t = '{} & {} & {} & {} & {}'.format(
				p['config']['db'],
				p['config']['batch_size'],
				p['config']['final_activation'].replace('poml', 'logit')
												 .replace('pomp', 'probit')
												 .replace('pomclog', 'c log-log'),
			'${:.0E}}}$'.format(p['config']['lr']).replace('E-0', '0^{-')
				.replace('E+0', '0^{+'),
				p['config']['activation']
			)

			train += t
			val += t
			test += t

			for metric, values in metrics['Train'].items():
				if metric != 'Confusion matrix':
					train += ' & ${:.5f}'.format(round(np.mean(values), 5))
					if show_std:
						train += '_{{({:.5f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 5))
					train += '$'

			train += '\\\\\n'

			for metric, values in metrics['Validation'].items():
				if metric != 'Confusion matrix':
					val += ' & ${:.5f}'.format(round(np.mean(values), 5))
					if show_std:
						val += '_{{({:.5f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 5))
					val += '$'

			val += '\\\\\n'

			for metric, values in metrics['Test'].items():
				if metric != 'Confusion matrix':
					test += ' & ${:.5f}'.format(round(np.mean(values), 5))
					if show_std:
						test += '_{{({:.5f})}}'.format(round(np.std(values, ddof=min(1, len(values)-1)), 5))
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
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	train_rs_op = K.image.resize_images(x_train, shape, method=K.image.ResizeMethod.BILINEAR)
	test_rs_op = K.image.resize_images(x_test, shape, method=K.image.ResizeMethod.BILINEAR)
	val_perc = 0.2

	with K.Session() as sess:
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

def create_retinopathy_h5(train_path, test_path, train_csv, test_csv, val_split, output_path):
	print('Loading train csv...')
	train_df = pd.read_csv(train_csv)
	print('Splitting train into train and validation...')
	train_df, val_df = train_test_split(train_df, test_size=val_split, random_state=1)
	print('Loading test csv...')
	test_df = pd.read_csv(test_csv)
	extension = '.jpeg'
	target_shape = (128, 128)

	datasets = {
		'train': {'name': 'training',
				  'df': train_df,
				  'path': train_path,
				  'output_name': 'retinopathy_128_train.h5',
				  'output_dir' : 'train'
				  },
		'val': {'name': 'validation',
				  'df': val_df,
				  'path': train_path,
				  'output_name': 'retinopathy_128_val.h5',
				  'output_dir' : 'val'
				  },
		'test': {'name': 'testing',
				  'df': test_df,
				  'path': test_path,
				  'output_name': 'retinopathy_128_test.h5',
				  'output_dir' : 'test'
				  }
	}

	for key, ds in datasets.items():
		print('--- Creating {} dataset ---'.format(ds['name']))
		x = []
		y = []

		# Create dir
		os.makedirs(os.path.join(output_path, ds['output_dir']))

		for i, (index, row) in enumerate(ds['df'].iterrows()):
			label = row['level']

			if not os.path.isdir(os.path.join(output_path, ds['output_dir'], str(label))):
				os.makedirs(os.path.join(output_path, ds['output_dir'], str(label)))

			# Avoid corrupt images
			try:
				f = open(os.path.join(ds['path'], row['image'] + extension), 'rb')
				file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
				img = cv2.imdecode(file_bytes, 1)
			except:
				continue

			# img = cv2.imread(os.path.join(ds['path'], row['image'] + extension))

			if not img is None:
				top, bottom, left, right = find_bbox(img)
				img = crop_img(img, top, bottom, left, right)
				img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
				cv2.imwrite(os.path.join(output_path, ds['output_dir'], str(label), row['image'] + extension), img)
				x.append(img)
				y.append(label)

				print('{}/{} {} loaded (class {})'.format(i, ds['df'].shape[0], row['image'], label))


		x = np.array(x)
		y = np.array(y)

		print('Standardizing channel pixels...')
		# Standardize each color channel
		means = x.mean(axis=(0, 1, 2))
		stds = x.std(axis=(0, 1, 2), ddof=1)
		x = (x - means) / stds

		print('Creating file {}...'.format(os.path.join(output_path, ds['output_name'])))
		with h5py.File(os.path.join(output_path, ds['output_name']), 'w') as f:
			f.create_dataset('x', data=x, compression='gzip', compression_opts=9)
			f.create_dataset('y', data=y, compression='gzip', compression_opts=9)

def split_train_val(trainval_path, output_path, val_split):

	if os.path.isdir(output_path):
		print("Output path already exists")
		return
	else:
		os.makedirs(output_path)

	for lbl in os.listdir(trainval_path):
		l = np.array(os.listdir(os.path.join(trainval_path, lbl)))
		train, val = train_test_split(l, test_size=val_split)

		os.makedirs(os.path.join(output_path, 'train', lbl))
		os.makedirs(os.path.join(output_path, 'val', lbl))

		for f in train:
			copy(os.path.join(trainval_path, lbl, f), os.path.join(output_path, 'train', lbl, f))

		for f in val:
			copy(os.path.join(trainval_path, lbl, f), os.path.join(output_path, 'val', lbl, f))



# # # Preprocess functions # # #

def find_bbox(img):
	top = 0
	left = 0
	right = img.shape[1]
	bottom = img.shape[0]

	# Find left
	for x in range(0, img.shape[1]):
		if img[:,x,:].mean() > 4.0:
			left = x
			break

	# Find right
	for x in reversed(range(0, img.shape[1])):
		if img[:, x, :].mean() > 4.0:
			right = x
			break

	# Find top
	for y in range(0, img.shape[0]):
		if img[y, :, :].mean() > 4.0:
			top = y
			break

	# Find bottom
	for y in reversed(range(0, img.shape[0])):
		if img[y, :, :].mean() > 4.0:
			bottom = y
			break

	return top, bottom, left, right

def crop_img(img, top, bottom, left, right):
	return img[top:bottom,left:right]


# # # Menu options # # #

def option_resume_one_metric():
	results_path = input('Results path: ')
	metric = input('Metric name: ')
	resume_one_metric(metric, results_path)

def option_resume_one_metric_csv():
	results_path = input('Results path: ')
	metric = input('Metric name: ')
	csv_path = input('Output CSV file path: ')
	resume_one_metric_csv(metric, results_path, csv_path)

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

def option_create_retinopathy_h5():
	train_path = input('Train path: ')
	test_path = input('Test path: ')
	train_csv = input('Train csv file: ')
	test_csv = input('Test csv file: ')
	val_split = float(input('Validation split ratio: '))
	output_path = input('Output path: ')
	create_retinopathy_h5(train_path, test_path, train_csv, test_csv, val_split, output_path)

def option_split_train_val():
	trainval_path = input('Trainval directory: ')
	output_path = input('Output path: ')
	val_split = float(input('Split percentage [0,1]: '))
	split_train_val(trainval_path, output_path, val_split)

def show_menu():
	print('=====================================')
	print('1. Resume results for one metric')
	print('2. Show confusion matrices')
	print('3. Show latex table')
	print('4. Create h5 dataset')
	print('5. Create h5 cifar10')
	print('6. Display images from h5')
	print('7. Create retinopathy h5')
	print('8. Split train and val')
	print('9. Resume results for one metric in csv file')
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
	elif option == '7':
		option_create_retinopathy_h5()
	elif option == '8':
		option_split_train_val()
	elif option == '9':
		option_resume_one_metric_csv()


if __name__ == '__main__':
	select_option(show_menu())
