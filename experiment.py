import tensorflow as tf
import numpy as np
import resnet
from net_keras import Net
import os
import time
import click
from scipy import io as spio
from callbacks import MomentumScheduler

class Experiment():
	def __init__(self, name, db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum):
		self.name = name
		self.db = db
		self.net_type = net_type
		self.batch_size = batch_size
		self.epochs = epochs
		self.checkpoint_dir = checkpoint_dir
		self.activation = activation
		self.spp_alpha = spp_alpha
		self.lr = lr
		self.momentum = momentum

		self.finish_callback = None

	def set_finish_callback(self, callback):
		self.finish_callback = callback

	def run(self):
		pass
