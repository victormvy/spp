# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import os
import prettytable

results_path = '../results_adience_h5'
evaluation_file = 'evaluation.pickle'


def resume_one_metric(metric):
	t = prettytable.PrettyTable(['Dataset', 'BS', 'LR', 'LF', 'Train ' + metric, 'Mean Tr', 'Validation ' + metric, 'Mean V', 'Test ' + metric, 'Mean Te'])
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

			if p:
				t.add_row([
					p['config']['db'],
					p['config']['batch_size'],
					p['config']['lr'],
					p['config']['final_activation'],
					train,
					train_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(train_values), 5), round(np.std(train_values), 5)) or 0,
					val,
					val_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(val_values), 5), round(np.std(val_values), 5)) or 0,
					test,
					test_values.size > 0 and '{:.5} ± {:.5}'.format(round(np.mean(test_values), 5), round(np.std(test_values), 5)) or 0
				])


	print(t)


def show_confusion_matrices():
	t = prettytable.PrettyTable(
		['Dataset', 'BS', 'LR', 'LF', 'Execution', 'Train CF', 'Validation CF', 'Test CF'], hrules=prettytable.ALL)
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
								item2,
								p['metrics']['Train']['Confusion matrix'],
								p['metrics']['Validation']['Confusion matrix'],
								p['metrics']['Test']['Confusion matrix']
							])

	print(t)


def show_latex_table():
	header, train, val, test = '', '', '', ''
	for item in sorted(os.listdir(results_path)):
		if os.path.isdir(os.path.join(results_path, item)):
			metrics = {'Train' : {}, 'Validation' : {}, 'Test' : {}}
			count = len(os.listdir(os.path.join(results_path, item)))
			for item2 in sorted(os.listdir(os.path.join(results_path, item))):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(
						os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)

						for metric, value in p['metrics']['Train'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Train']:
									metrics['Train'][metric] += value / count
								else:
									metrics['Train'][metric] = value / count

						for metric, value in p['metrics']['Validation'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Validation']:
									metrics['Validation'][metric] += value / count
								else:
									metrics['Validation'][metric] = value / count

						for metric, value in p['metrics']['Test'].items():
							if metric != 'Confusion matrix':
								if metric in metrics['Test']:
									metrics['Test'][metric] += value / count
								else:
									metrics['Test'][metric] = value / count
			
			if header == '':
				header = 'Dataset & BS & LR & LF'

				for metric, value in metrics['Train'].items():
					if metric != 'Confusion matrix':
						header += ' & {}'.format(metric)

				header += '\\\\\\hline'

			t = '{} & {} & {} & {}'.format(
				p['config']['db'],
				p['config']['batch_size'],
				p['config']['lr'],
				p['config']['final_activation'])

			train += t
			val += t
			test += t

			for metric, value in metrics['Train'].items():
				if metric != 'Confusion matrix':
					train += ' & {:.5f}'.format(round(value, 5))

			train += '\\\\\n'

			for metric, value in metrics['Validation'].items():
				if metric != 'Confusion matrix':
					val += ' & {:.5f}'.format(round(value, 5))

			val += '\\\\\n'

			for metric, value in metrics['Test'].items():
				if metric != 'Confusion matrix':
					test += ' & {:.5f}'.format(round(value, 5))

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


def option_resume_one_metric():
	metric = input('Metric name: ')
	resume_one_metric(metric)


def show_menu():
	print('=====================================')
	print('1. Resume results for one metric')
	print('2. Show confusion matrices')
	print('3. Show latex table')
	print('=====================================')
	option = input(' Choose one option: ')

	return option


def select_option(option):
	if option == '1':
		option_resume_one_metric()
	elif option == '2':
		show_confusion_matrices()
	elif option == '3':
		show_latex_table()


if __name__ == '__main__':
	select_option(show_menu())
