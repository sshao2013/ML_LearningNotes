import numpy as np


def pca(data, k):
    n_samples, n_features = data.shape
    mean = np.array([np.mean(data[:, i]) for i in range(n_features)])
    normalization_result = data - mean
    scatter_matrix = np.dot(np.transpose(normalization_result), normalization_result) / n_samples
    print(scatter_matrix)
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    print(eig_val)
    print(eig_vec)
    eig_result = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_result.sort(reverse=True)
    feature = np.array([ele[1] for ele in eig_result[:k]])
    final_result = np.dot(normalization_result, np.transpose(feature))
    return final_result


data = np.array([[1, 1], [1, 3], [2, 3], [4, 4], [2, 4]])
print(pca(data, 1))
