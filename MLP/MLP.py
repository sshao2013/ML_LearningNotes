import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    # sigmoid function
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def load_dataset(filename):
    fr = open(filename)
    x = []
    y = []
    for line in fr.readlines():
        line = line.strip().split()
        x.append([float(line[0]), float(line[1])])
        y.append([float(line[-1])])
    return np.mat(x), np.mat(y)


def forward_propagation(x, w1, b1, w2, b2):
    z1 = x * w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 * w2 + b2
    a2 = sigmoid(z2)
    return a1, a2


def backward_propagation(x, y, w1, b1, w2, b2, a1, a2, learning_rate):
    a0 = x.copy()
    delta2 = a2 - y
    delta1 = np.mat((delta2 * w2.T).A * (a1.A * (1 - a1).A))
    dw1 = a0.T * delta1
    db1 = np.sum(delta1, 0)
    dw2 = a1.T * delta2
    db2 = np.sum(delta2, 0)

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    return w1, b1, w2, b2


def gradient_descent(x, y, h_dim, num_iterations, learning_rate):
    # x is (20,2) then w1 is (2,3) 3 is hidden layer dimension
    w1 = np.mat(np.random.rand(2, h_dim))
    b1 = np.mat(np.random.rand(1, h_dim))
    # the input for output is from hidden layer(20,3), w is (3,1)
    w2 = np.mat(np.random.rand(h_dim, 1))
    b2 = np.mat(np.random.rand(1, 1))

    for i in range(num_iterations):
        a1, a2 = forward_propagation(x, w1, b1, w2, b2)
        w1, b1, w2, b2 = backward_propagation(x, y, w1, b1, w2, b2, a1, a2, learning_rate)
    return w1, b1, w2, b2


def scaling(x):
    x_max = np.max(x, 0)
    x_min = np.min(x, 0)
    return (x - x_min) / (x_max - x_min), x_max, x_min


def run():
    learning_rate = 0.01
    num_iterations = 20000
    hidden_layer_dimension = 10

    x, y = load_dataset('plot_data.txt')
    x_scaling, x_max, x_min = scaling(x)

    w1, b1, w2, b2 = gradient_descent(x_scaling, y, hidden_layer_dimension, num_iterations, learning_rate)

    # graph part
    plotx1 = np.arange(0, 10, 0.01)
    plotx2 = np.arange(0, 10, 0.01)
    plotX1, plotX2 = np.meshgrid(plotx1, plotx2)
    plotx_new = np.c_[plotX1.ravel(), plotX2.ravel()]  # transform data to (n,2)
    plotx_new2 = (plotx_new - x_min) / (x_max - x_min)  # testdata scaling
    plt.clf()

    # get boundary
    plot_z1 = plotx_new2 * w1 + b1
    plot_a1 = sigmoid(plot_z1)
    plot_z2 = plot_a1 * w2 + b2
    plot_a2 = sigmoid(plot_z2)

    ploty_new = np.reshape(plot_a2, plotX1.shape)  # transform data shape

    plt.contourf(plotX1, plotX2, ploty_new, 1, alpha=0.5)
    plt.scatter(x[:, 0][y == 0].A, x[:, 1][y == 0].A)
    plt.scatter(x[:, 0][y == 1].A, x[:, 1][y == 1].A)
    plt.show()


if __name__ == '__main__':
    run()
