import numpy as np
import operator

dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']


def KNN(target, dataset, labels, k):
    data_size = dataset.shape[0]
    diff = np.tile(target, (data_size, 1)) - dataset
    sqr_data = diff ** 2
    sqr_data_result = sqr_data.sum(axis=1)
    distances = sqr_data_result ** 0.5
    sort_result = distances.argsort()
    count = {}
    for i in range(k):
        vote_result = labels[sort_result[i]]
        count[vote_result] = count.get(vote_result, 0) + 1
    final_result = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return final_result[0][0]


print(KNN([1.1, 1.1], dataset, labels, 3))
