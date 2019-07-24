import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import tensorflow as tf


def sigmoid(x):
    # sigmoid function
    h = 1.0 / (1.0 + np.exp(-x))
    return h


def create_dataset(size):
    X, y = make_classification(n_samples=size, n_classes=2, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    ones = np.ones([size, 1])
    X = np.hstack([ones, X])

    return X, y


def run():
    learning_rate = 0.1
    num_iterations = 100
    size = 100

    x, y_raw = create_dataset(size)
    y = y_raw.reshape(size, 1)
    m, n = x.shape
    print(x.shape, y.shape)

    X = tf.placeholder(tf.float32, [None, n])
    Y = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([n, 1]))

    h = tf.nn.sigmoid(tf.matmul(X, W))
    cost = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=h, labels=Y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_iterations):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        Weight = sess.run(W)

    print(Weight)

    # boundary function
    # 0 = w0 + w1x1 + w2x2
    w0 = Weight[0, 0]
    w1 = Weight[1, 0]
    w2 = Weight[2, 0]

    x1 = np.linspace(-5, 5)
    x2 = - w0 / w2 - w1 / w2 * x1

    plt.scatter(x[:, 1], x[:, 2], c=y_raw, cmap="RdBu")
    plt.plot(x1, x2)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()


if __name__ == '__main__':
    run()
