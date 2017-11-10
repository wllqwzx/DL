import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S") 
root_logdir = "tf_logs" 
logdir = "{}/run-{}/".format(root_logdir, now)


def make_relu(threshold):
    THRESHOLD = tf.Variable(threshold, name="threshold")    
    # to make THRESHOLD visiable in tensorboard, it should have the type tf.Variable 
    # rather than a tf.constant or just numberical value.
    def relu(X):
        with tf.name_scope("relu"):
            W_shape = (int(X.get_shape()[1]) ,1)
            w = tf.Variable(tf.random_normal(W_shape), name="weights")
            b = tf.Variable(0.0, name="bias")
            z = tf.add(tf.matmul(X, w), b, name="z")
            return tf.maximum(z, THRESHOLD, name="relu")
    return relu # THRESHOLD will be a outside variable for relu, but local variabel for top level

relu = make_relu(0.0)

X = tf.placeholder(tf.float32, shape=(None, 3), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")


file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(output, feed_dict={X:[[1,2,3]]})

file_writer.close()