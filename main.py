import tensorflow as tf
import numpy as np
import resnet
from net import Net
import os
import time

train, test = tf.keras.datasets.cifar10.load_data()
train_x, train_y_cls = train
test_x, test_y_cls = test

train_x = train_x / 255.0
test_x = test_x / 255.0

num_classes = 10

#train_y = np.zeros((len(train_y_cls), 10))
#train_y[np.arange(10), train_y_cls] = 1
train_y = np.eye(num_classes)[train_y_cls].reshape([len(train_y_cls),num_classes])
test_y = np.eye(num_classes)[test_y_cls].reshape([len(test_y_cls),num_classes])

#np.set_printoptions(threshold=np.nan)
#print(train_y)

train = None
test = None

batch_size = 128
epochs = 100
checkpoint_dir = 'model'
log_dir = 'model'
checkpoint_file = 'model.ckpt'


x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x')
y_true = tf.placeholder(dtype=tf.int32, shape=[None, num_classes], name='y_true')

#net = resnet.inference(x, 9, False)
#vgg = Vgg19()
#net = vgg.build(x)
#net = tf.keras.applications.VGG19(include_top=False, input_tensor=x, input_shape=(32, 32, 3), classes=num_classes)
net_object = Net(32, 'spp', 3, num_classes)
net_object.spp_alpha = 0.8
net = net_object.vgg19(x)

y_pred = tf.nn.softmax(net)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_true_cls = tf.argmax(y_true, axis=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.5, momentum=0.1, use_nesterov=True).minimize(cost)

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

	for epoch in range(start_epoch, epochs+1):
		tStart = time.time()
		current = 0
		mean_train_acc = 0
		num_batches = 0
		while current < 50000:	
			_, acc = sess.run([optimizer, accuracy], feed_dict= {
				x: train_x[current:current+batch_size],
				y_true: train_y[current:current+batch_size]
			})

			print("Epoch {}/{}. Images from {} to {}. Accuracy: {}".format(epoch, epochs, current, current+batch_size, acc))			
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
				x: test_x[current:current+batch_size],
				y_true: test_y[current:current+batch_size]
			})
			mean_test_acc += test_acc
			current = current + batch_size
		
		mean_test_acc /= num_batches

		if mean_test_acc > best_test_acc:
			best_test_acc = mean_test_acc

		tElapsed = time.time() - tStart

		print("End of epoch {}. Train accuracy: {}, Test accuracy: {}, Time: {}s".format(epoch, mean_train_acc, mean_test_acc, tElapsed))

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