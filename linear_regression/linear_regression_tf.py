import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

size = 50
mu, sigma = 0, 0.1  # mean and standard deviation


def create_dataset():
    init_w = 5
    init_b = 10
    # y = wx + b + noise
    x = np.linspace(-1, 1, size)
    noise = np.random.normal(mu, sigma, size)
    y = init_w * x + init_b + noise
    plt.scatter(x, y)
    return x, y


def run():
    x, y = create_dataset()
    num_iterations = 1000
    learning_rate = 0.01
    cost_log = np.array([])

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(np.random.randn(), name="W")
    b = tf.Variable(np.random.randn(), name="b")

    # Hypothesis
    h = tf.add(tf.multiply(X, W), b)

    # MSE
    cost = tf.reduce_sum(tf.pow(h - Y, 2)) / (2 * size)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_iterations):
            for (_x, _y) in zip(x, y):
                sess.run(optimizer, feed_dict={X: _x, Y: _y})

            # log the cost
            c = sess.run(cost, feed_dict={X: x, Y: y})
            cost_log = np.append(cost_log, c)

        weight = sess.run(W)
        bias = sess.run(b)

    plt.subplot(121)
    plt.plot(np.linspace(1, num_iterations, num_iterations, endpoint=True), cost_log)
    plt.title("Result")
    plt.xlabel("Iterations")
    plt.ylabel("Cost History")

    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, weight * x + bias)
    plt.show()

    print('final w:', weight)
    print('final b:', bias)
    print('final hypothesis: ', 'y = ', weight, 'x + ', bias)


if __name__ == "__main__":
    run()
