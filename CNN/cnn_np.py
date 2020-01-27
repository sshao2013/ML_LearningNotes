import numpy as np
import mnist


class Convlayer:  # 3x3
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.rand(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backprop(self, d_out, learn_rate):
        d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_filters[f] += d_out[i, j, f] * im_region

            # Update filters
        self.filters -= learn_rate * d_filters
        return None


class MaxPool:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))  # array max
        return output

    def backprop(self, d_out):
        d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_input[i * 2 + i2, j * 2 + j2, f2] = d_out[i, j, f2]

        return d_input


class Softmax:  # fully connected layer
    def __init__(self, input_len, nodes):
        self.weights = np.random.rand(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_out, learn_rate):
        for i, gradient in enumerate(d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            d_sum = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (d_sum ** 2)
            d_out_d_t[i] = t_exp[i] * (d_sum - t_exp[i]) / (d_sum ** 2)
            dw = self.last_input
            dbias = 1
            d_inputs = self.weights
            d_loss = gradient * d_out_d_t
            dw = dw[np.newaxis].T @ d_loss[np.newaxis]  # matrix multiplication
            dbias = d_loss * dbias
            d_inputs = d_inputs @ d_loss

            self.weights -= learn_rate * dw
            self.biases -= learn_rate * dbias

            return d_inputs.reshape(self.last_input_shape)


class CNN:
    def __init__(self):
        self.conv = Convlayer(8)  # 28x28x1 -> 26x26x8
        self.pool = MaxPool()  # 26x26x8 -> 13x13x8
        self.softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10

    def forward(self, image, label):
        out = self.conv.forward(image / 255)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, im, lable, learn_rate=0.01):
        out, loss, acc = self.forward(im, label)
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]
        gradient = self.softmax.backprop(gradient, learn_rate)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, learn_rate)
        return loss, acc


train_images = mnist.train_images()[:2000]
train_labels = mnist.train_labels()[:2000]
test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]

cnn = CNN()

for epoch in range(2):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 0:
            print(
                'Average Loss %.3f | Accuracy: %d%%' %
                (loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = cnn.train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = cnn.forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
