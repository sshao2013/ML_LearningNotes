import tensorflow as tf
import numpy as np


def binary_generation(int2binary, numbers):
    binary_x = np.array([int2binary[num] for num in numbers], dtype=np.uint8)
    binary_x = np.fliplr(binary_x)
    return binary_x


def batch_generation(int2binary, batch_size, largest_number):
    n1 = np.random.randint(0, largest_number // 2, batch_size)
    n2 = np.random.randint(0, largest_number // 2, batch_size)
    add = n1 + n2

    # int to binary
    binary_n1 = binary_generation(int2binary, n1)
    binary_n2 = binary_generation(int2binary, n2)
    batch_y = binary_generation(int2binary, add)

    batch_x = np.dstack((binary_n1, binary_n2))

    return batch_x, batch_y, n1, n2, add


def binary2int(binary_array):
    out = 0
    for index, x in enumerate(reversed(binary_array)):
        out += x * pow(2, index)
    return out


def run():
    batch_size = 64
    hidden_size = 20

    int2bin = {}
    bin_dim = 8
    max_num = pow(2, bin_dim)
    binary = np.unpackbits(np.array([range(max_num)], dtype=np.uint8).T, axis=1)

    for i in range(max_num):
        int2bin[i] = binary[i]

    x = tf.placeholder(tf.float32, [None, bin_dim, 2])
    y = tf.placeholder(tf.float32, [None, bin_dim])

    cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    W = tf.Variable(tf.truncated_normal([hidden_size, 1], stddev=0.01))
    b = tf.zeros([1])

    predictions = tf.reshape(tf.sigmoid(tf.matmul(outputs, W) + b), [-1, bin_dim])

    cost = tf.losses.mean_squared_error(y, predictions)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            input_x, input_y, _, _, _ = batch_generation(int2bin, batch_size, max_num)
            sess.run(optimizer, feed_dict={x: input_x, y: input_y})

        val_x, val_y, n1, n2, add = batch_generation(int2bin, batch_size, max_num)
        result = sess.run(predictions, feed_dict={x: val_x, y: val_y})
        result = np.fliplr(np.round(result))
        result = result.astype(np.int32)

        for b_x, b_p, a, b, add in zip(np.fliplr(val_x), result, n1, n2, add):
            print('{} + {} = {}'.format(a, b, add))
            print('{}:{}'.format(b_p, binary2int(b_p)))
            print("------------")


if __name__ == '__main__':
    run()
