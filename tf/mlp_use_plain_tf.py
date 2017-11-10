import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs" 
logdir = "{}/run-{}/".format(root_logdir, now)


mnist = input_data.read_data_sets("/tmp/data/")


#======
n_input = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10


X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_input = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_input)
        init_W = tf.truncated_normal((n_input,n_neurons), mean=0.0, stddev=stddev)
        W = tf.Variable(initial_value=init_W, name="Weights")
        b = tf.Variable(initial_value=tf.zeros(n_neurons), name="biases")
        z = tf.matmul(X,W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", "relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", "relu")
    logits = neuron_layer(hidden2, n_output, "outputs")

'''
#tensorflow provide the same function as neuron_layer we just defined,
#and it take cares more things like regulizations. To use it, write the following code:

from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1") 
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2") 
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
'''

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


test_accu_summary = tf.summary.scalar("Test Accu", accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#======

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for i in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

        accu_train = sess.run(accuracy, feed_dict={X:X_batch,y:y_batch})
        accu_test = sess.run(accuracy, feed_dict={X:mnist.test.images, y:mnist.test.labels})
        print("train accuracy:", accu_train, "test accrracy:", accu_test)

        summary_str = test_accu_summary.eval(feed_dict={X:mnist.test.images, y:mnist.test.labels})
        file_writer.add_summary(summary_str, epoch)

    savepath = saver.save(sess, "./my_model/my_model_final.ckpt")

file_writer.close()

