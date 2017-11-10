import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs" 
logdir = "{}/run-{}/".format(root_logdir, now)


# without name_scope
def relu1(X):
    W_shape = (int(X.get_shape()[1]) ,1)
    w = tf.Variable(tf.random_normal(W_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

# with name_scope "relu2"
def relu2(X):
    with tf.name_scope("relu2"):
        W_shape = (int(X.get_shape()[1]) ,1)
        w = tf.Variable(tf.random_normal(W_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")


# X1 = tf.placeholder(tf.float32, shape=(None, 3), name="X1")
# relu1s = [relu1(X1) for i in range(5)]
# output1 = tf.add_n(relu1s, name="output1")

X2 = tf.placeholder(tf.float32, shape=(None, 3), name="X2")
relu2s = [relu2(X2) for i in range(5)]
output2 = tf.add_n(relu2s, name="output2")


file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(output2, feed_dict={X2:[[1,2,3]]})

file_writer.close()
'''
name_scope is extremely important when you want to visuliz the
computing graph in the tensorboard.
'''
