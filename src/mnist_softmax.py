import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Building model
y = tf.nn.softmax(tf.matmul(x, W) + b) #[None,10] shape

# Placeholder for cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])

#difference between model distribution and actual distribution
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Optimization used during automatic backpropogation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Running training steps while introducing small batches of random data (stochastic training)
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#returns boolean array
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Convert boolean array to int array and take mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
