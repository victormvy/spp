import os
import json
import tensorflow as tf
from experiment import Experiment

class ExperimentSet():
	def __init__(self, experiments=[]):
		self._experiments = experiments

	# PROPERTIES

	@property
	def experiments(self):
		return self._experiments

	@experiments.setter
	def experiments(self, experiments):
		self._experiments = experiments

	@experiments.deleter
	def experiments(self):
		del self._experiments

	# # # # # #

	def _validate_experiments(self):
		if not type(self.experiments) is list:
			if type(self.experiments) is tuple:
				self.experiments = list(self.experiments)
			else:
				self.experiments = []

	def add_experiment(self, experiment):
		self._validate_experiments()
		self.experiments.append(experiment)

	def remove_experiment(self, name):
		self._validate_experiments()
		for experiment in self.experiments:
			if experiment.name == name:
				self.experiments.remove(experiment)

	def clear_experiments(self):
		self.experiments = []

	def load_from_file(self, path):
		with open(path) as f:
			configs = json.load(f)

		for config in configs:
			if 'executions' in config and config['executions'] > 1:
				for execution in range(1, int(config['executions']) + 1):
					exec_config = config.copy()
					if 'name' in exec_config:
						exec_config['name'] += "_{}".format(execution)
					exec_config['checkpoint_dir'] += "/{}".format(execution)
					experiment = Experiment()
					experiment.set_config(exec_config)
					self.add_experiment(experiment)

			elif not 'executions' in config or ('executions' in config and config['executions'] > 0):
				experiment = Experiment()
				experiment.set_config(config)
				self.add_experiment(experiment)

	def save_to_file(self, path):
		configs = []

		for experiment in self.experiments:
			configs.append(experiment.get_config())

		json.dump(configs, path)


	def run_all(self, gpu_number=0):
		for experiment in self.experiments:
			if not experiment.finished:
				os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
				with tf.device('/device:GPU:' + str(gpu_number)):
					experiment.run()
					# Clear session
					tf.keras.backend.clear_session()
