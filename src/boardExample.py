import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape, name):
	return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name);

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
	# add layer name scopes for better graph visual
	with tf.name_scope("layer1"):
		X = tf.nn.dropout(X,p_keep_input)
		h = tf.nn.relu(tf.matmul(X,w_h))
	with tf.name_scope("layer2"):
		h 	= tf.nn.dropout(h,p_keep_hidden)
		h2 	= tf.nn.relu(tf.matmul(h, w_h2))
	with tf.name_scope("layer3"):
		h2 	= tf.nn.dropout(h2, p_keep_hidden)
		return tf.matmul(h2, w_o)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# create placeholders
X = tf.placeholder("float", [None, 784], name= "X")
Y = tf.placeholder("float", [None, 10], name="Y")

# init weights
w_h = init_weights([784, 625], "w_h")
w_h2 = init_weights([625,625], "w_h2")
w_o = init_weights([625,10], "w_o")

# adding the dropouts' place holders
p_keep_input = tf.placeholder("float", name="p_keep_input")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

# creating model
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

# create cost (negative scoring) function (measure)
with tf.name_scope("cost"):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
	# adding scalar summary for cost tensor
	tf.summary.scalar("cost", cost)

# Measuring accuracy
with tf.name_scope("accuracy"):
	# Num of accurate predictions
	correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(py_x,1)) 
	# cast boolean tensor to float then take average
	acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))
	# add scalar summary for accuracy tensor
	tf.summary.scalar("accuracy", acc_op)

# creating session
with tf.Session() as sess:
	# creating log writter. run 'tensorboard --logdir=./logs/nn_logs'
	writter = tf.summary.FileWriter("./logs/nn_logs", sess.graph) 
	merged = tf.summary.merge_all()

	# initializing model
	tf.initialize_all_variables().run()

	# train the model
	for i in range(15):

		for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
			sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
											p_keep_input:  0.08, p_keep_hidden:0.5})

		summary, acc = sess.run([merged, acc_op], feed_dict={X:teX, Y:teY,
								p_keep_input: 1.0, p_keep_hidden: 1.0})
		writter.add_summary(summary, i) #writting summary
		print(i, acc) #printing the accuracy





