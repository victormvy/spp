import json
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
		configs = json.load(path)

		for config in configs:
			experiment = Experiment()
			experiment.set_config(config)
			self.add_experiment(experiment)

	def save_to_file(self, path):
		configs = []

		for experiment in self.experiments:
			configs.append(experiment.get_config())

		json.dump(configs, path)
