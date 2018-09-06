import tensorflow as tf
import numpy as np
import resnet
from net import Net
import os
import time
import click
from scipy import io as spio


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
@click.option('--rep', '-r', default=1, help=u'Repetitions for this execution.')
def main(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum, rep):
	train(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum, rep)

def train(db, net_type, batch_size, epochs, checkpoint_dir, log_dir, activation, spp_alpha, lr, momentum, rep):
	train = None
	test = None	
	num_channels = 3	
	img_size = 32
	
	if db == '10':
		train, test = tf.keras.datasets.cifar10.load_data()
		num_classes = 10
	elif db == '100':
		train, test = tf.keras.datasets.cifar100.load_data()
		num_classes = 100
	elif db == 'emnist':
		emnist = spio.loadmat('emnist/emnist-byclass.mat')
		
		train_x = np.reshape(emnist['dataset'][0][0][0][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
		train_y_cls = emnist['dataset'][0][0][0][0][0][1]
		
		test_x = np.reshape(emnist['dataset'][0][0][1][0][0][0], (-1, 28, 28, 1)).astype(np.float32)
		test_y_cls = emnist['dataset'][0][0][1][0][0][1]
		
		num_classes = 62
		num_channels = 1
		img_size = 28
		
	else:
		print("Invalid database. Database must be 10 or 100")

	if train:
		train_x, train_y_cls = train
	train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls), num_classes])
	
	if test:
		test_x, test_y_cls = test
	test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls), num_classes])

	train_x = train_x / 255.0
	test_x = test_x / 255.0
	
	train = None
	test = None

	checkpoint_file = 'model.ckpt'

	x = tf.placeholder(dtype=tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
	y_true = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='y_true')

	if net_type == 'resnet56':
		net = resnet.inference(x, 9, False)
	elif net_type == 'resnet110':
		net = resnet.inference(x, 18, False)
	elif net_type == 'vgg19':
		net_object = Net(32, activation, num_channels, num_classes)
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

	lrt = tf.placeholder(dtype=tf.float32, name='lr')

	cost = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.MomentumOptimizer(learning_rate=lrt, momentum=momentum, use_nesterov=True).minimize(cost)

	saver = tf.train.Saver()
	
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)	
		
	start_epoch = 1
	best_test_acc = 0.0
	start_execution = 1
	
	# Read file of current epoch
	if os.path.isfile(os.path.join(checkpoint_dir, 'current_epoch.txt')):
		with open(os.path.join(checkpoint_dir, 'current_epoch.txt'), 'r') as f:
			start_epoch = int(f.readline())
			best_test_acc = float(f.readline())
			start_execution = int(f.readline())
	
	results = {'best_test_acc': []}

	for execution in range(start_execution, rep + 1):
		execution_checkpoint = '{}/{}'.format(checkpoint_dir, execution)
		execution_log = '{}/{}'.format(log_dir, execution)
		with tf.Session() as sess:
			
			sess.run(tf.global_variables_initializer())
	
			if os.path.isfile(os.path.join(checkpoint_dir, "checkpoint")):
				try:
					saver.restore(sess, os.path.join(execution_checkpoint, checkpoint_file))
					print("Restored checkpoint")
				except tf.errors.NotFoundError:
					print("No restore point found")

			
			writer = tf.summary.FileWriter(execution_log, sess.graph)
			
			curr_lr = lr
			
			for epoch in range(start_epoch, epochs + 1):		
				tStart = time.time()
				
				# Reduce learning rate in epochs 60, 80 and 90.
				if epoch in { 60, 80, 90 }:
					if curr_lr > 0.02:
						curr_lr -= 0.02
						print('Learning rate reduced ({})'.format(curr_lr))
					else:
						print('Learning rate could not be reduced ({})'.format(curr_lr))
				
				current = 0
				mean_train_acc = 0
				num_batches = 0
				while current < len(train_x):
					_, acc = sess.run([optimizer, accuracy], feed_dict={
						x: train_x[current:current + batch_size],
						y_true: train_y[current:current + batch_size],
						lrt: curr_lr
					})
	
					print("Epoch {}/{}. Images from {} to {} of {}. Accuracy: {}".format(epoch, epochs, current,
																				   current + batch_size, len(train_x), acc))
					current = current + batch_size
					mean_train_acc += acc
					num_batches += 1
	
				mean_train_acc /= num_batches
	
				# Test accuracy
				mean_test_acc = 0
				current = 0
				num_batches = 0
				while current < len(test_x):
					num_batches += 1
					test_acc = sess.run(accuracy, feed_dict={
						x: test_x[current:current + batch_size],
						y_true: test_y[current:current + batch_size],
						lrt: curr_lr
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
	
				# Save current epoch and execution in file
				with open(os.path.join(checkpoint_dir, 'current_epoch.txt'), 'w') as f:
					f.write("{}\n{}\n{}".format(str(epoch + 1), str(best_test_acc), str(execution + 1)))
	
			# Save result
			results['best_test_acc'].append(best_test_acc)
		start_epoch = 1
		best_test_acc = 0.0
	results['test_acc_mean'] = np.mean(results['best_test_acc'])
	results['test_acc_var'] = np.var(results['best_test_acc'])
	print(results)
	return results

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
									print("RUNNING {}".format(dirname))
									train(db, net_type, bs, epochs, dirname, dirname, activation, spp_alpha, lr, momentum)
									tf.reset_default_graph()


if __name__ == '__main__':
	cli()
