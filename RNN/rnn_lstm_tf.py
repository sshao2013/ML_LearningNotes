import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def RNN(x, W, b, input_num, sequence_length, hidden_num):
    input_x = tf.reshape(x, [-1, sequence_length, input_num])
    cust_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
    output, final_state = tf.nn.dynamic_rnn(cust_lstm_cell, input_x, dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(final_state[1], W) + b)


def run():
    mnist = input_data.read_data_sets("./data", one_hot=True)

    batch_size = 200
    batch_num = mnist.train.num_examples // batch_size

    input_num = 28
    sequence_length = 28
    hidden_num = 100
    class_num = 10

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.truncated_normal([hidden_num, class_num], stddev=0.1))
    b = tf.Variable(tf.zeros([class_num]))

    rnn_model = RNN(x, W, b, input_num, sequence_length, hidden_num)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=rnn_model, labels=y))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    pred = tf.equal(tf.argmax(rnn_model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20):
            for batch in range(batch_num):
                x_data, y_data = mnist.test.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x: x_data, y: y_data})
            train_result = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
            test_result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Round " + str(i) + " Train Result: " + str(train_result) + " Test Result: " + str(test_result))


if __name__ == '__main__':
    run()
