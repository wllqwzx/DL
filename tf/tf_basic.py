import tensorflow as tf

# 1.create the computing graph
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# 2.create a session
sess = tf.Session()

# 3.initilize the variable
sess.run(x.initializer)
sess.run(y.initializer)

# 4.evaluates
results = sess.run(f)
print(results)

# 5.close the session
sess.close()


# simpler approach
#===========================
# 1.create the computing graph
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# 2.create global_variables_initializer
init = tf.global_variables_initializer()

# 3.create a session and run
with tf.Session() as sess: 
    init.run()  # initialize all the variables
    result = f.eval()