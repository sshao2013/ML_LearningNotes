import copy
import numpy as np

np.random.seed(0)


# ref: https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/


def sigmoid(x):
    # sigmoid function
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_dev(z):
    return z * (1 - z)


def create_data(max_num):
    int2bin = {}
    # store the binary code 0-256
    binary = np.unpackbits(np.array([range(max_num)], dtype=np.uint8).T, axis=1)
    for i in range(max_num):
        int2bin[i] = binary[i]

    return int2bin


def forward_propagation(x_bin, y_bin, z_bin, result, U, V, W, hidden_dim, bin_dim, hidden_layer_values, out_delta,
                        error_log):
    for position in range(bin_dim):
        X = np.array([[x_bin[bin_dim - position - 1], y_bin[bin_dim - position - 1]]])
        y = np.array([z_bin[bin_dim - position - 1]]).T

        hidden_layer = sigmoid(np.dot(X, U) + np.dot(hidden_layer_values[-1], W))
        out_layer = sigmoid(np.dot(hidden_layer, V))
        out_layer_error = y - out_layer
        out_delta.append(out_layer_error * sigmoid_dev(out_layer))
        error_log += np.abs(out_layer_error[0])

        # to int
        result[bin_dim - position - 1] = np.round(out_layer[0][0])

        # save to mem
        hidden_layer_values.append(copy.deepcopy(hidden_layer))

    future_hidden_layer_delta = np.zeros(hidden_dim)
    return future_hidden_layer_delta, out_delta, error_log


def backward_propagation(x_bin, y_bin, U, V, W, V_update, W_update, U_update, bin_dim, hidden_layer_values, out_delta,
                         learning_rate,
                         future_hidden_layer_delta):
    for position in range(bin_dim):
        X = np.array([[x_bin[position], y_bin[position]]])
        hidden_layer = hidden_layer_values[-position - 1]
        prev_hidden_layer = hidden_layer_values[-position - 2]

        # error at output layer
        out_layer_delta = out_delta[-position - 1]
        # error at hidden layer
        hidden_layer_delta = (future_hidden_layer_delta.dot(W.T) + out_layer_delta.dot(
            V.T)) * sigmoid_dev(hidden_layer)
        V_update += np.atleast_2d(hidden_layer).T.dot(out_layer_delta)
        W_update += np.atleast_2d(prev_hidden_layer).T.dot(hidden_layer_delta)
        U_update += X.T.dot(hidden_layer_delta)

        future_hidden_layer_delta = hidden_layer_delta

    U += U_update * learning_rate
    V += V_update * learning_rate
    W += W_update * learning_rate

    U_update *= 0
    V_update *= 0
    W_update *= 0
    return U, V, W, future_hidden_layer_delta


def predict(i, error_log, z, result, x_int, y_int):
    if i % 1000 == 0:
        print("Error:" + str(error_log))
        print("Pred:" + str(result))
        print("True:" + str(z))
        out = 0
        for index, x in enumerate(reversed(result)):
            out += x * pow(2, index)
        print(str(x_int) + " + " + str(y_int) + " = " + str(out))
        print("------------")


def run():
    learning_rate = 0.1
    input_dim = 2  # two number to plus, dim is 2
    hidden_dim = 20
    output_dim = 1
    bin_dim = 8
    max_num = pow(2, bin_dim)  # max is 256 here
    num_iterations = 10000

    int2bin = create_data(max_num)

    # keep the weight from -1 to 1
    U = 2 * np.random.random((input_dim, hidden_dim)) - 1
    V = 2 * np.random.random((hidden_dim, output_dim)) - 1
    W = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

    U_update = np.zeros_like(U)
    V_update = np.zeros_like(V)
    W_update = np.zeros_like(W)

    for i in range(num_iterations):
        # addition x + y = z
        x_int = np.random.randint(max_num / 2)
        x_bin = int2bin[x_int]  # get binary
        y_int = np.random.randint(max_num / 2)
        y_bin = int2bin[y_int]
        z_int = x_int + y_int
        z_bin = int2bin[z_int]

        result = np.zeros_like(z_bin)  # predict result

        error_log = 0

        out_delta = list()
        hidden_layer_values = list()
        hidden_layer_values.append(np.zeros(hidden_dim))

        future_hidden_layer_delta, out_delta, error_log = forward_propagation(x_bin, y_bin, z_bin, result, U, V, W, hidden_dim, bin_dim,
                                                        hidden_layer_values, out_delta, error_log)

        U, V, W, future_hidden_layer_delta = backward_propagation(x_bin, y_bin, U, V, W, V_update, W_update, U_update,
                                                                  bin_dim, hidden_layer_values, out_delta,
                                                                  learning_rate,
                                                                  future_hidden_layer_delta)
        predict(i, error_log, z_bin, result, x_int, y_int)


if __name__ == '__main__':
    run()
