import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


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


def gradient_descent(X, y, learning_rate, size, num_iterations):
    w = np.mat(np.random.rand(3, 1))
    y = y.reshape(size, 1)
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, w))
        dw = np.dot(X.T, (h - y))
        w = w - learning_rate * dw

    return w


def run():
    learning_rate = 0.1
    num_iterations = 100
    size = 100

    X, y = create_dataset(size)
    w = gradient_descent(X, y, learning_rate, size, num_iterations)

    # boundary function
    # 0 = w0 + w1x1 + w2x2
    w0 = w[0, 0]
    w1 = w[1, 0]
    w2 = w[2, 0]

    x1 = np.linspace(-5, 5)
    x2 = - w0 / w2 - w1 / w2 * x1

    plt.scatter(X[:, 1], X[:, 2], c=y, cmap="RdBu")
    plt.plot(x1,x2)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()
    print(w)


if __name__ == '__main__':
    run()
