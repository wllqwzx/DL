import tensorflow as tf
import numpy as np
import pandas as pd

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal([n_inputs,n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal([n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1,n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0,Wx) + b)
Y1 = tf.tanh(tf.matmul(X1,Wx) + tf.matmul(Y0,Wy) + b)


#=====
init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0 
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1],feed_dict={X0:X0_batch, X1:X1_batch})
    print(Y0_val)
    print(Y1_val)




#===============
# use tf static_rnn ro unroll an RNN
XX0 = tf.placeholder(tf.float32, [None, n_inputs])
XX1 = tf.placeholder(tf.float32, [None, n_inputs])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [XX0,XX1], dtype = tf.float32)

YY0, YY1 = output_seqs


