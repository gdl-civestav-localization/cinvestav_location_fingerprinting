import numpy as np
import matplotlib.pyplot as plt
import math

__author__ = 'Gibran'


class Perceptron:

    W = np.array([1, 1, 1])
    errors = []

    def __init__(self, W):
        self.W = W

    def activation_fuction(self, v):
        if v >= 0:
            return 1
        else:
            return 0

    def error_fuction(self, y, d):
        e = d - y
        return e

    def norm(self, v):
        return math.fabs(v)

    def train(self, training_data, eta, theta):
        i = 0
        while True:
            i += 1
            totalerror = 0
            for data in training_data:

                X = data[0]
                d = data[1]

                v = np.dot(self.W, X)
                y = self.activation_fuction(v)

                e = self.error_fuction(y, d)

                delta_w = eta * e * np.array(X)
                self.W = self.W + delta_w
                totalerror += self.norm(e)

            self.errors.append(totalerror)
            if totalerror <= theta or i >= 1000:
                print self.W
                break

    def test(self, data):
        X = data
        v = np.dot(self.W, X)
        y = self.activation_fuction(v)

        return y


def plot_error(neuron):
    plt.plot(neuron.errors)
    plt.axis('equal')
    plt.xlabel("iterations")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


def plot_data(dataset, W=[1, 1, 0]):
    X = np.array([])
    Y = np.array([])
    D = np.array([])
    for data in dataset:
        x = data[0]
        d = data[1]
        X = np.append(X, x[1])
        Y = np.append(Y, x[2])
        D = np.append(D, d)

    rgb = plt.get_cmap('jet')(2 * D)
    plt.scatter(X, Y, color=rgb, linewidths=8)

    m = W[2] / W[1]
    x = np.arange(0, neuron.W[1], 0.1)
    y = m * x + 0
    plt.plot(x, y)

    if m != 0:
        x2 = np.arange(0, 1, 0.1)
        m2 = -1 / m
        y2 = m2 * x2 + W[0]
        plt.plot(x2, y2)

    plt.grid()
    plt.show()


if __name__ == "__main__":

    training_data = [
        [[1, 0, 0], 0],
        [[1, 0, 1], 0],
        [[1, 1, 0], 0],
        [[1, 1, 1], 1]
    ]

    W = np.array([1, 1, 1])  # Initial Weight
    theta = 0.01  # Stop Criteria
    eta = 0.1  # Learning rate

    neuron = Perceptron(W)
    neuron.train(training_data, eta, theta)

    test_data = [
        [1, 0, 3],
        [1, .2, 1],
        [1, .6, .5],
        [1, 1.1, 1]
    ]

    test_data = np.random.random((4000, 3))
    test_data[:, 0] = 1.0

    results = []
    for data in test_data:
        result = neuron.test(data)
        results.append([np.array(data), result])

    plot_data(training_data)
    plot_error(neuron)
    plot_data(results, W)
