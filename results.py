import numpy as np
import pandas as pd
import pickle
import os
from prettytable import PrettyTable

results_path = '../results_testing'
evaluation_file = 'evaluation.pickle'

def resume_one_metric(metric):
	t = PrettyTable(['Dataset', 'BS', 'LR', 'LF', 'Train ' + metric, 'Validation ' + metric, 'Test ' + metric])
	for item in os.listdir(results_path):
		if os.path.isdir(os.path.join(results_path, item)):
			for item2 in os.listdir(os.path.join(results_path, item)):
				if os.path.isdir(os.path.join(results_path, item, item2)) and os.path.isfile(os.path.join(results_path, item, item2, evaluation_file)):
					with open(os.path.join(results_path, item, item2, evaluation_file), 'rb') as f:
						p = pickle.load(f)
						t.add_row([
							p['config']['db'],
							p['config']['batch_size'],
							p['config']['lr'],
							p['config']['final_activation'],
							round(p['metrics']['Train'][metric], 5),
							round(p['metrics']['Validation'][metric], 5),
							round(p['metrics']['Test'][metric], 5)
						])

	print(t)

def option_resume_one_metric():
	metric = input('Metric name: ')
	resume_one_metric(metric)

def show_menu():
	print('=====================================')
	print('1. Resume results for one metric')
	print('=====================================')
	option = input(' Choose one option: ')

	return option

def select_option(option):
	if option == '1':
		option_resume_one_metric()


if __name__ == '__main__':
	select_option(show_menu())