import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

size = 50
mu, sigma = 0, 0.1  # mean and standard deviation


def create_dataset():
    init_w = 5
    init_b = 10
    # y = wx + b + noise
    x = np.linspace(-1, 1, size)
    noise = np.random.normal(mu, sigma, size)
    y = init_w * x + init_b + noise
    return x, y


def run():
    plt.figure(figsize=(10, 5))
    # init the dataset
    x, y = create_dataset()

    x = np.reshape(x, (-1, 1))

    linear_regression = sklearn.linear_model.LinearRegression()
    linear_regression.fit(x, y)

    w = linear_regression.coef_
    b = linear_regression.intercept_

    plt.scatter(x, y)
    plt.plot(x, w * x + b)
    plt.show()

    print('final w:', w)
    print('final b:', b)
    print('final hypothesis: ', 'y = ', w[0], 'x + ', b)


if __name__ == '__main__':
    run()
