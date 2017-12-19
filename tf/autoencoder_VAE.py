import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, shape=[28,28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


#===
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")

X_test = mnist.test.images
y_test = mnist.test.labels.astype("int")

n_inputs = 28*28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20

n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.001

with tf.contrib.framework.arg_scope(
    [fully_connected], 
    activation_fn=tf.nn.elu, 
    weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
    
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)

    hidden3_mean = fully_connected(hidden2, n_hidden3, activation_fn=None)
    hidden3_gamma = fully_connected(hidden2, n_hidden3, activation_fn=None)
    hidden3_sigma = tf.exp(0.5*hidden3_gamma)
    noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
    hidden3 = hidden3_mean + hidden3_sigma*noise

    hidden4 = fully_connected(hidden3, n_hidden4)
    hidden5 = fully_connected(hidden4, n_hidden5)
    logits = fully_connected(hidden5, n_outputs, activation_fn=None)
    
    outputs = tf.sigmoid(logits)


reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
latent_loss = 0.5*tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

n_epochs = 5
n_digits = 60
batch_size = 150

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
            if iteration % 50 == 0:
                print("epoch: ", epoch, " batch:", iteration, " loss: ", sess.run(loss, feed_dict={X:X_batch}))
        
    sample_codings = np.random.normal(size = [n_digits, n_hidden3])
    sample_outputs = sess.run(outputs, feed_dict={hidden3: sample_codings}) #!!! hidden3 depends on X


for iteration in range(n_digits):
    plt.subplot(n_digits/10, 10, iteration+1)
    plot_image(sample_outputs[iteration])
    #plt.imsave("img"+str(iteration)+".jpg", sample_outputs[iteration].reshape([28,28]), cmap="Greys")
plt.show()
