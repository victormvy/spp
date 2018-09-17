import tensorflow as tf
import numpy as np
import resnet
from net_keras import Net
import os
import time
import click
from scipy import io as spio
from callbacks import MomentumScheduler


@click.group()
def cli():
	pass


@cli.command('train', help='Train model')
@click.option('--db', default='10', help=u'Database that will be used: Cifar10 (10), Cifar100 (100) or EMNIST.')
@click.option('--net_type', '-n', default='vgg19',
			  help=u'Net model that will be used. Must be one of: vgg19, resnet56, resnet110')
@click.option('--batch_size', '-b', default=128, help=u'Batch size')
@click.option('--epochs', '-e', default=100, help=u'Number of epochs')
@click.option('--checkpoint_dir', '-d', required=True, help=u'Checkpoint files directory')
@click.option('--activation', '-a', default='relu', help=u'Activation function')
@click.option('--spp_alpha', default=0.2, help=u'Alpha value for spp transfer function')
@click.option('--lr', default=0.1, help=u'Learning rate')
@click.option('--momentum', '-m', default=0.1, help=u'Momentum for optimizer')
@click.option('--rep', '-r', default=1, help=u'Repetitions for this execution.')
def main(db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum, rep):
	train(db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum, rep)

def train(db, net_type, batch_size, epochs, checkpoint_dir, activation, spp_alpha, lr, momentum, rep):
	num_channels = 3
	img_size = 32
	
	if db == '10':
		(train_x, train_y_cls), (test_x, test_y_cls) = tf.keras.datasets.cifar10.load_data()
		num_classes = 10
	elif db == '100':
		(train_x, train_y_cls), (test_x, test_y_cls) = tf.keras.datasets.cifar100.load_data()
		num_classes = 100
	elif db.lower() == 'emnist':
		emnist = spio.loadmat('emnist/emnist-byclass.mat')
		
		train_x = np.reshape(emnist['dataset'][0][0][0][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
		train_y_cls = emnist['dataset'][0][0][0][0][0][1]
		
		test_x = np.reshape(emnist['dataset'][0][0][1][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
		test_y_cls = emnist['dataset'][0][0][1][0][0][1]
		
		num_classes = 62
		num_channels = 1
		img_size = 28
		
	else:
		print("Invalid database. Database must be 10, 100 or EMNIST")
		return

	train_x = train_x / 255.0
	test_x = test_x / 255.0

	train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls), num_classes])
	test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls), num_classes])


	def learning_rate_scheduler(epoch):
		final_lr = lr
		if epoch >= 60:
			final_lr -= 0.02
		if epoch >= 80:
			final_lr -= 0.02
		if epoch >= 90:
			final_lr -= 0.02
		return float(final_lr)

	def momentum_scheduler(epoch):
		final_mmt = momentum
		if epoch >= 60:
			final_mmt -= 0.0005
		if epoch >= 80:
			final_mmt -= 0.0005
		if epoch >= 90:
			final_mmt -= 0.0005
		return float(final_mmt)

	def save_epoch(epoch, logs):
		with open(os.path.join(checkpoint_dir, model_file_extra), 'w') as f:
			f.write(str(epoch+1))

	save_epoch_callback = tf.keras.callbacks.LambdaCallback(
		on_epoch_end=save_epoch
	)

	if net_type == 'resnet56':
		#net = resnet.inference(x, 9, False)
		raise NotImplementedError
	elif net_type == 'resnet110':
		#net = resnet.inference(x, 18, False)
		raise NotImplementedError
	elif net_type == 'vgg19':
		net_object = Net(img_size, activation, num_channels, num_classes, spp_alpha)
		net_object.spp_alpha = spp_alpha
		model = net_object.vgg19()
	else:
		print('Invalid net type. You must select one of these: vgg19, resnet56, resnet110')
		return

	if not os.path.isdir(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(db, net_type, batch_size, activation, spp_alpha, lr,
											   momentum, rep)

	for execution in range(1, rep + 1):
		if not os.path.isdir(os.path.join(checkpoint_dir, model_name)):
			os.makedirs(os.path.join(checkpoint_dir, model_name))

		model_file = "{}/model.h5py".format(model_name)
		model_file_extra = "{}/model.txt".format(model_name)
		csv_file = "{}/results.csv".format(model_name)

		start_epoch = 0


		if os.path.isfile(os.path.join(checkpoint_dir, model_file)) and os.path.isfile(os.path.join(checkpoint_dir, model_file_extra)):
			print("===== RESTORING SAVED MODEL =====")
			model.load_weights(os.path.join(checkpoint_dir, model_file))

			with open(os.path.join(checkpoint_dir, model_file_extra), 'r') as f:
				start_epoch = int(f.readline())

		model.compile(
			optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
			loss = 'categorical_crossentropy',
			metrics = ['accuracy']
		)

		model.summary()


		model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, initial_epoch=start_epoch,
				  callbacks=[ tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
							  MomentumScheduler(momentum_scheduler),
							  tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, model_file)),
							  save_epoch_callback,
							  tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_dir, csv_file)),
							  tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir)
							  ],
				  validation_data=(test_x, test_y)
				  )



# def main(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum):

@cli.command('experiment', help='Train model with different set of parameters')
@click.option('--path', '-p', required=True, help=u'Directory where results will be stored.')
def experiment(path):
	dbs = ['10','100']
	net_types = ['vgg19']
	bss = [128]
	epochs = 100
	activations = ['relu','lrelu','elu','softplus','spp']
	spp_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	lrs = [0.1, 0.01, 0.001]
	momentums = [0.9, 0.5, 0.1]


	for db in dbs:
		for net_type in net_types:
			for bs in bss:
				for activation in activations:
					if activation == 'spp':
						_spp_alphas = spp_alphas
					else:
						_spp_alphas = [0]
					for spp_alpha in _spp_alphas:
						for lr in lrs:
							for momentum in momentums:
									dirname = "{}/{}_{}_{}_{}_{}_{}_{}".format(path, db, net_type, bs, activation, spp_alpha, lr,
																			momentum)
									print("==================================")
									print("RUNNING DB: {}\nNET: {}\nBATCH SIZE: {}\nACTIVATION: {}\nSPP ALPHA: {}\nLEARNING RATE: {}\nMOMENTUM: {}".format(db, net_type, bs ,activation, spp_alpha, lr, momentum))
									print("==================================")
									train(db, net_type, bs, epochs, dirname, activation, spp_alpha, lr, momentum, 10)
									tf.reset_default_graph()


if __name__ == '__main__':
	cli()
