from experiment import Experiment

experiment = Experiment('test', '10', 'vgg19', 128, 2, 'test', 'relu', 0, 1e-4, 0.9)
experiment.set_auto_name()
experiment.checkpoint_dir = "test/{}".format(experiment.get_auto_name())
experiment.run()
