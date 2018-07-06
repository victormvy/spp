import tensorflow as tf
import numpy as np
import resnet
from net import Net
import os
import time
import click


@click.group()
def cli():
	pass


@cli.command('train', help='Train model')
@click.option('--db', default='10', help=u'Database that will be used: Cifar10 (10) or Cifar100 (100).')
@click.option('--net_type', '-n', default='vgg19',
			  help=u'Net model that will be used. Must be one of: vgg19, resnet56, resnet110')
@click.option('--batch_size', '-b', default=128, help=u'Batch size')
@click.option('--epochs', '-e', default=100, help=u'Number of epochs')
@click.option('--checkpoint_dir', '-d', required=True, help=u'Checkpoint files directory')
@click.option('--log_dir', '-l', required=True, help=u'Log files directory')
@click.option('--activation', '-a', default='relu', help=u'Activation function')
@click.option('--spp_alpha', default=0.2, help=u'Alpha value for spp transfer function')
@click.option('--lr', default=0.1, help=u'Learning rate')
@click.option('--momentum', '-m', default=0.1, help=u'Momentum for optimizer')
def main(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum):
	train(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum)

def train(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum):
	if db == '10':
		train, test = tf.keras.datasets.cifar10.load_data()
		num_classes = 10
	elif db == '100':
		train, test = tf.keras.datasets.cifar100.load_data()
		num_classes = 100
	else:
		print("Invalid database. Database must be 10 or 100")

	train_x, train_y_cls = train
	test_x, test_y_cls = test

	train_x = train_x / 255.0
	test_x = test_x / 255.0

	train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls), num_classes])
	test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls), num_classes])

	train = None
	test = None

	checkpoint_file = 'model.ckpt'

	x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x')
	y_true = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='y_true')

	if net_type == 'resnet56':
		net = resnet.inference(x, 9, False)
	elif net_type == 'resnet110':
		net = resnet.inference(x, 18, False)
	elif net_type == 'vgg19':
		net_object = Net(32, activation, 3, num_classes)
		net_object.spp_alpha = spp_alpha
		net = net_object.vgg19(x)
	else:
		print('Invalid net type. You must select one of these: vgg19, resnet56, resnet110')

	y_pred = tf.nn.softmax(net)
	y_pred_cls = tf.argmax(y_pred, axis=1)
	y_true_cls = tf.argmax(y_true, axis=1)

	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_true)

	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=True).minimize(cost)

	saver = tf.train.Saver()

	print(net)

	with tf.Session() as sess:
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		sess.run(tf.global_variables_initializer())

		if os.path.isfile(os.path.join(checkpoint_dir, "checkpoint")):
			try:
				saver.restore(sess, os.path.join(checkpoint_dir, checkpoint_file))
				print("Restored checkpoint")
			except tf.errors.NotFoundError:
				print("No restore point found")

		start_epoch = 1
		best_test_acc = 0.0

		# Read file of current epoch
		if os.path.isfile(os.path.join(checkpoint_dir, 'current_epoch.txt')):
			with open(os.path.join(checkpoint_dir, 'current_epoch.txt'), 'r') as f:
				start_epoch = int(f.readline())
				best_test_acc = float(f.readline())

		writer = tf.summary.FileWriter(log_dir, sess.graph)

		for epoch in range(start_epoch, epochs + 1):
			tStart = time.time()
			current = 0
			mean_train_acc = 0
			num_batches = 0
			while current < 50000:
				_, acc = sess.run([optimizer, accuracy], feed_dict={
					x: train_x[current:current + batch_size],
					y_true: train_y[current:current + batch_size]
				})

				print("Epoch {}/{}. Images from {} to {}. Accuracy: {}".format(epoch, epochs, current,
																			   current + batch_size, acc))
				current = current + batch_size
				mean_train_acc += acc
				num_batches += 1

			mean_train_acc /= num_batches

			# Test accuracy
			mean_test_acc = 0
			current = 0
			num_batches = 0
			while current < 10000:
				num_batches += 1
				test_acc = sess.run(accuracy, feed_dict={
					x: test_x[current:current + batch_size],
					y_true: test_y[current:current + batch_size]
				})
				mean_test_acc += test_acc
				current = current + batch_size

			mean_test_acc /= num_batches

			if mean_test_acc > best_test_acc:
				best_test_acc = mean_test_acc

			tElapsed = time.time() - tStart

			print("End of epoch {}. Train accuracy: {}, Test accuracy: {}, Time: {}s".format(epoch, mean_train_acc,
																							 mean_test_acc, tElapsed))

			summary = tf.Summary()
			summary.value.add(tag="Train accuracy", simple_value=mean_train_acc)
			summary.value.add(tag="Test accuracy", simple_value=mean_test_acc)
			writer.add_summary(summary, epoch)
			print("Summary saved!")

			# Save current epoch in file
			with open(os.path.join(checkpoint_dir, 'current_epoch.txt'), 'w') as f:
				f.write(str(epoch + 1))
				f.write('\n')
				f.write(str(best_test_acc))


# def main(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum):

@cli.command('experiment', help='Train model with different set of parameters')
def experiment():
	dbs = ['10']
	net_types = ['vgg19']
	bss = [128]
	epochs = 10
	activations = ['spp']
	spp_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	lrs = [0.1]
	momentums = [0.9]

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
								dirname = "{}_{}_{}_{}_{}_{}_{}".format(db, net_type, bs, activation, spp_alpha, lr,
																		momentum)
								print("RUNNING {}".format(dirname))
								train(db, net_type, bs, epochs, dirname, dirname, activation, spp_alpha, lr, momentum)


if __name__ == '__main__':
	cli()
