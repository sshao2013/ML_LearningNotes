import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def run():
    # load data from tensorflow
    mnist = input_data.read_data_sets('data/', one_hot=True)

    batch_size = 200
    batch_num = mnist.train.num_examples // batch_size

    input_size = 784
    output_size = 10
    hidden_dim = 100

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])

    W_L1 = tf.Variable(tf.random_normal([input_size, hidden_dim],stddev=0.1))
    b_L1 = tf.Variable(tf.random_normal([hidden_dim],stddev=0.1))
    L1 = tf.nn.sigmoid(tf.matmul(x, W_L1) + b_L1)

    W_L2 = tf.Variable(tf.random_normal([hidden_dim, output_size],stddev=0.1))
    b_L2 = tf.Variable(tf.random_normal([output_size],stddev=0.1))
    L2 = tf.nn.softmax(tf.matmul(L1, W_L2) + b_L2)

    cost = tf.nn.softmax_cross_entropy_with_logits(logits=L2, labels=y)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    pred = tf.equal(tf.arg_max(y, 1), tf.argmax(L2, 1))  # return bool
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))  # cast bool to float

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            for batch in range(batch_num):
                x_data, y_data = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: x_data, y: y_data})
            train_result = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
            test_result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Round " + str(i) + " Train Result: " + str(train_result)+ " Test Result: " + str(test_result))


if __name__ == '__main__':
    run()
