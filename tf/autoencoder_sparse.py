import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images

def kl(p,q):
    return p*tf.log(p/q) + (1-p)*tf.log((1-p)/(1-q))

def get_batch(data, batch, batch_size):
    length = len(data)
    start = batch*batch_size
    end = (batch+1)*batch_size
    if end > length:
        end = length
    return data[start:end]

#=====
n_input = 28*28
n_hidden = 1000     #!!! sparse aitoencoder need code to be over-complete!!!
n_output = n_input

learning_rate = 0.01
target_sparsity = 0.1
sparsity_weight = 0.2

X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.sigmoid)
logits = fully_connected(hidden, n_output, activation_fn = None)
output = tf.nn.sigmoid(logits)

hidden_mean = tf.reduce_mean(hidden, axis=1) # mean over one image
sparse_loss = tf.reduce_sum(kl(target_sparsity, hidden_mean))
reconstruction_loss = tf.reduce_mean(tf.square(output-X))
loss = reconstruction_loss + sparsity_weight*sparse_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


n_epochs = 50
batch_size = 150

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size + 1
        for batch in range(n_batches):
            X_batch = get_batch(X_train, batch, batch_size)
            sess.run(training_op, feed_dict={X:X_batch})
        recons_loss, spar_loss, tot_loss = sess.run([reconstruction_loss, sparse_loss, loss], feed_dict={X:X_train})
        print("epochs:", epoch, "recons loss:", recons_loss, 
              "sparse loss:", spar_loss, "loss:", tot_loss)

