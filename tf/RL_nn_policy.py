import gym
import numpy as np
import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 1
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(inputs=X, units=n_hidden, activation=tf.nn.elu)
logit = tf.layers.dense(inputs=hidden, units=1, activation=None)

output = tf.nn.sigmoid(logit)

p_left_right = tf.concat(axis=1, values=[output, 1 - output])

action = tf.multinomial(tf.log(p_left_right), num_samples=1)
y = 1.0 - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grad_and_var = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, var in grad_and_var]


init = tf.global_variables_initializer()

