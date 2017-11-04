import tensorflow as tf
import numpy as np 
import sklearn
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing() 
m, n = housing.data.shape 
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]


n_epochs = 1000 
learning_rate = 0.01

# transform the data to let all of its fratures' mean=0 and std=1
transformer = sklearn.preprocessing.StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = transformer.transform(housing_data_plus_bias)

# create node and computing graph
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X") 
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")      # total loss

# compute the gradients and manually optimize
#=============================
# gradients = 2/m * tf.matmul(tf.transpose(X), error)   # Manually Computing the Gradients
gradients = tf.gradients(mse, [theta])[0]               # computing the gradients with autodiff
                                                        # the mes has only one output, so the index is 0
training_op = tf.assign(theta, theta - learning_rate * gradients) 


# use a optimizer
#=============================
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)
'''

# eval the graph in a session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("epoch:", epoch, " MSE=", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print(best_theta)
