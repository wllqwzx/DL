import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from data_loader import get_batch

mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
y_train = mnist.train.labels.astype("int")
X_test = mnist.test.images
y_test = mnist.test.labels.astype("int")

#===
X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
Y = tf.placeholder(dtype=tf.int32, shape=[None])

hidden = X
hidden = tf.layers.dense(hidden, 256, activation=tf.nn.relu)
hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu)
hidden = tf.layers.dense(hidden, 10, activation=None)
logits = hidden

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(xentropy)

correct = tf.nn.in_top_k(logits, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#========
learning_rate = 0.001
trainable_vars = xs=tf.trainable_variables()    # get all of the trainable variables in current graph
var_grad_list = tf.gradients(ys=loss, xs=trainable_vars)    # compute gradient of loss w.r.t all trainable vars
var_count = len(var_grad_list)

update = [] # use update to store all of the weight update operation nodes
for index in range(var_count):
    var = trainable_vars[index]
    var_grad = var_grad_list[index]
    update.append(var.assign(var - learning_rate*var_grad))
#========


n_epoch = 50
batch_size = 150

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epoch):
        n_batch = X_train.shape[0] // batch_size
        for batch in range(n_batch):
            X_batch = get_batch(X_train, batch, batch_size)
            Y_batch = get_batch(y_train, batch, batch_size)
            sess.run(update, feed_dict={X:X_batch, Y:Y_batch})  # here we run the update, it contains all of
                                                                # the assignment operations for each variables
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X:X_train, Y:y_train})
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict={X:X_test, Y:y_test})
        print("epoch", epoch, "train loss:", train_loss, "train accuracy", train_acc, 
              "test loss:", test_loss, "test accuracy:", test_acc)

