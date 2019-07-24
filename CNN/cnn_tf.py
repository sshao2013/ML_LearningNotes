import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def cnn_weight(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.1))


def cnn_bias(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.1))


def cnn_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def cnn_max_pool(h_conv):
    return tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def run():
    # load data from tensorflow
    mnist = input_data.read_data_sets('data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, height, weight, channel]

    W_conv_1 = cnn_weight([5, 5, 1, 32])
    b_conv_1 = cnn_bias([32])
    h_conv_1 = tf.nn.relu(cnn_conv2d(x_image, W_conv_1) + b_conv_1)
    h_pool_1 = cnn_max_pool(h_conv_1)

    W_conv_2 = cnn_weight([5, 5, 32, 64])
    b_conv_2 = cnn_bias([64])
    h_conv_2 = tf.nn.relu(cnn_conv2d(h_pool_1, W_conv_2) + b_conv_2)
    h_pool_2 = cnn_max_pool(h_conv_2)

    W_fc_1 = cnn_weight([7 * 7 * 64, 1024])
    b_fc_1 = cnn_bias([1024])
    h_pool_2_reshape = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
    h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_reshape, W_fc_1) + b_fc_1)

    prob = tf.placeholder(tf.float32)
    h_fc_1_drop = tf.nn.dropout(h_fc_1, prob)

    W_fc_2 = cnn_weight([1024, 10])
    b_fc_2 = cnn_bias([10])
    h_fc_2 = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_fc_2, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
    pred = tf.equal(tf.argmax(h_fc_2, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    batch_size = 100
    batch_num = mnist.train.num_examples // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20):
            for batch in range(batch_num):
                x_data, y_data = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: x_data, y: y_data, prob: 0.8})
            train_result = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, prob:1.0})
            test_result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, prob:1.0})
            print("Round " + str(i) + " Train Result: " + str(train_result) + " Test Result: " + str(test_result))


if __name__ == '__main__':
    run()
