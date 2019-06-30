import numpy as np
import random
import os, struct
from array import array as pyarray

# ref: http://neuralnetworksanddeeplearning.com/chap1.html


def sigmoid(z):
    # sigmoid function
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def forward_propagation(x, B, W):
    for b, w in zip(B, W):
        x = sigmoid(np.dot(w, x) + b)
    return x


def backward_propagation(x, y, num_layers, W, B):
    nabla_b = [np.zeros(b.shape) for b in B]
    nabla_w = [np.zeros(w.shape) for w in W]

    activation = x
    activations = [x]
    zs = []
    for b, w in zip(B, W):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    delta = cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(W[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    return (nabla_b, nabla_w)


def update_mini_batch(B, W, num_layers, mini_batch, learning_rate):
    nabla_b = [np.zeros(b.shape) for b in B]
    nabla_w = [np.zeros(w.shape) for w in W]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backward_propagation(x, y, num_layers, W, B)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    W = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(W, nabla_w)]
    B = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(B, nabla_b)]

    return B, W


def gradient_descent(train_set, epoch, batch_size, learning_rate, test_set=None):
    if test_set:
        n_test = len(test_set)

    input_size = 28 * 28
    output_size = 10
    Hid_Layer = 40
    sizes = [input_size, Hid_Layer, output_size]
    num_layers = len(sizes)  # nn layer numbers
    W = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    B = [np.random.randn(y, 1) for y in sizes[1:]]

    n = len(train_set)
    for j in range(epoch):
        random.shuffle(train_set)
        mini_batches = [train_set[k:k + batch_size] for k in range(0, n, batch_size)]
        for mini_batch in mini_batches:
            B, W = update_mini_batch(B, W, num_layers, mini_batch, learning_rate)
        if test_set:
            print("Epoch {0}: {1} / {2}".format(j, evaluate(test_set, B, W), n_test))
        else:
            print("Epoch {0} complete".format(j))

    return B, W


def evaluate(test_data, B, W):
    test_results = [(np.argmax(forward_propagation(x, B, W)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)


def cost_derivative(output_activations, y):
    return output_activations - y


def predict(data, B, W):
    value = forward_propagation(data, B, W)
    return value.tolist().index(max(value))


def load_mnist(dataset="training_data", digits=np.arange(10), path="."):
    if dataset == "training_data":
        fname_image = os.path.join(path, 'train-images-idx3-ubyte')
        fname_label = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing_data":
        fname_image = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_label = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def load_samples(dataset="training_data"):
    image, label = load_mnist(dataset)

    X = [np.reshape(x, (28 * 28, 1)) for x in image]
    X = [x / 255.0 for x in X]

    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e

    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


def run():
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')

    B, W = gradient_descent(train_set, 10, 128, 5, test_set=test_set)

    correct = 0;
    for test_feature in test_set:
        if predict(test_feature[0], B, W) == test_feature[1][0]:
            correct += 1
    print("Accuracy: ", correct / len(test_set))


if __name__ == '__main__':
    run()
