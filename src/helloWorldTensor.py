import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(32)
b = tf.constant(10)

print(sess.run(a+b))
sess.close()