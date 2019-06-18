import numpy as np
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


def gradient_descent(b, w, x, y, learning_rate):
    # hypothesis: y = wx + b
    b = b - learning_rate * (1.0 / size) * ((w * x + b - y).sum(axis=0))
    w = w - learning_rate * (1.0 / size) * (((w * x + b - y) * x).sum(axis=0))
    return b, w


def cost_mean_squared_error(b, w, x, y):
    # hypothesis: y = wx + b
    y0 = w * x + b
    mse = (1.0/(2.0*size))*((np.square(y0 - y)).sum(axis=0))
    return mse


def run():
    # init the dataset
    learning_rate = 0.1
    b = np.random.rand(1) #init b value to start finding final value
    w = np.random.rand(1) #init w value to start finding final value
    num_iterations = 50

    plt.figure(figsize=(10, 5))

    x, y = create_dataset()

    cost_log = np.array([])

    for i in range(num_iterations):
        b, w = gradient_descent(b, w, x, y, learning_rate)
        cost_log = np.append(cost_log, cost_mean_squared_error(b, w, x, y))

    plt.subplot(121)
    plt.plot(np.linspace(1, num_iterations, num_iterations, endpoint=True), cost_log)
    plt.title("Result")
    plt.xlabel("Iterations")
    plt.ylabel("Cost History")

    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, w* x + b)
    plt.show()

    print('final w:', w)
    print('final b:', b)
    print('final hypothesis: ', 'y = ', w[0], 'x + ', b[0])


if __name__ == '__main__':
    run()
