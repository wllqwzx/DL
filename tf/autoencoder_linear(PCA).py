import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
X_test = mnist.test.images

n_input = 784
n_hidden = 128
n_output = n_input

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_input])
encoder = fully_connected(X, n_hidden, activation_fn = None)
decoder = fully_connected(encoder, n_output, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(decoder - X))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for e_poches in range(20):
        sess.run(training_op, feed_dict={X:X_train})
        training_mse = sess.run(reconstruction_loss, feed_dict={X:X_train})
        test_mse = sess.run(reconstruction_loss, feed_dict={X:X_test})
        print("epoch: ", e_poches, " training mse: ", training_mse, " test mse: ", test_mse)
    
