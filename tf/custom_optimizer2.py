import tensorflow as tf

data1 = [[1,2]]
data2 = [
    [1,2],
    [3,4],
    [5,6]
]

X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
W = tf.Variable(initial_value=[1.0,1.0])
Z = X*W + W

grad = tf.gradients(ys=Z, xs=W)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(sess.run(grad, feed_dict={X:data1}))  #[2,3]
    print(sess.run(grad, feed_dict={X:data2}))  #[12,15] !!!
                                                # if we put a data batch but a single data, the tf.gradients 
                                                # will compute all of the gradent for each data, and return 
                                                # the sum not the average of all the data