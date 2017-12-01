import tensorflow as tf
from tensorflow.contrib.layers import fully_connected   #!!!
from tensorflow.contrib.layers import batch_norm        #!!!
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs" 
logdir = "{}/run-{}/".format(root_logdir, now)

mnist = input_data.read_data_sets("/tmp/data/")



n_input = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_output = 10

#======
X = tf.placeholder(tf.float32, shape=(None,n_input), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# step 1
is_training = tf.placeholder(tf.bool, shape=(), name='is_training') 

# step 2
bn_params = {   
    'is_training' : is_training,
    'decay' : 0.99,
    'updates_collections' : None
}

# step 3
hidden1 = fully_connected(X, n_hidden1, scope="hidden1",normalizer_fn=batch_norm, normalizer_params=bn_params)
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", normalizer_fn=batch_norm, normalizer_params=bn_params)
logits = fully_connected(hidden2, n_output, scope="output", normalizer_fn=batch_norm, normalizer_params=bn_params)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1, name="correct")
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
            sess.run(training_op, feed_dict={is_training:True,X:X_batch, y:y_batch}) # need to saaign value to is_training
        
        accu_train = sess.run(accuracy, feed_dict={is_training:False,X:X_batch,y:y_batch})
        accu_test = sess.run(accuracy, feed_dict={is_training:False,X:mnist.test.images, y:mnist.test.labels})
        print("train accuracy:", accu_train, "test accrracy:", accu_test)
        summary_str = test_accu_summary.eval(feed_dict={is_training:False,X:mnist.test.images, y:mnist.test.labels})
        file_writer.add_summary(summary_str, epoch)

    savepath = saver.save(sess, "./my_model/my_model_final.ckpt")

file_writer.close()
