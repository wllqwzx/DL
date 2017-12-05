import tensorflow as tf


#===
n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_outputs])  # seq to seq

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

#==== fully connected layer of output for seq to seq rnn!!!
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.contrib.layers.fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
#====

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for e_poch in range(1000):
        X_batch, y_batch = [...] # fetch next batch
        sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        if e_poch % 10 == 0:
            mse = sess.run(loss, feed_dict={X:X_batch, y:y_batch})
            print(e_poch, ": mse: ", mse)


