import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def create_dataset(size):
    X, y = make_classification(n_samples=size, n_classes=2, n_features=2,
                               n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu")

    return X, y


def run():
    size = 100
    X, y = create_dataset(size)

    clf = LogisticRegression().fit(X, y)
    w = clf.coef_[0]

    alpha = -w[0]/w[1]
    plot_x = np.linspace(-5, 5)
    plot_y = alpha * plot_x - (clf.intercept_[0]) / w[1]
    plt.scatter(X[:,0],X[:,1],c=y,cmap="RdBu")
    plt.plot(plot_x, plot_y)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()
    print(w)


if __name__ == '__main__':
    run()
