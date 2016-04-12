import numpy as np
from scipy import random, zeros, exp, dot
from scipy.linalg import norm, inv, pinv

from cluster import Clustering

__author__ = 'Gibran'

random.seed(17)


class RBF:
    def __init__(self, input_length, hidden_length, out_lenght):
        self.input_length = input_length
        self.out_lenght = out_lenght
        self.hidden_length = hidden_length
        self.centers = []
        for i in xrange(hidden_length):
            self.centers.append(random.uniform(-1, 1, input_length))
        self.variance = 1
        self.W = random.random((self.hidden_length, self.out_lenght))

    def basisfunction(self, x, xi):
        assert len(xi) == self.input_length
        return exp(-1.0 / (2 * self.variance) * norm(x - xi) ** 2)

    def calc_activation(self, dataset):
        # calculate activations of RBFs
        green_matrix = zeros((len(dataset), self.hidden_length), float)
        for xi, x in enumerate(dataset):
            for ci, c in enumerate(self.centers):
                green_matrix[xi, ci] = self.basisfunction(x, c)
        return green_matrix

    def train(self, dataset, targets, lamda=0):
        """ dataset: matrix of dimensions n x input
            targets: column vector of dimension n x output """

        # Choose random center vectors from training set
        self.centers = Clustering.get_cluster_centers(dataset, self.hidden_length)
        # self.centers = random.permutation(dataset)[:self.hidden_length]

        # Calculate data variance
        self.variance = np.var(dataset)

        # Calculate activations of RBFs
        green_matrix = self.calc_activation(dataset)

        # Calculate output weights
        if lamda == 0:
            self.W = dot(pinv(green_matrix), targets)  # With pseudoinverse
        else:
            green_matrix_transpose = np.transpose(green_matrix)
            # With operator lambda
            self.W = dot(inv(dot(green_matrix_transpose, green_matrix) + lamda * np.identity(self.hidden_length)),
                         dot(green_matrix_transpose, targets))

        # Get error
        result = self.test(dataset)
        error = self.cost_function(targets, result)
        return error

    def cost_function(self, targets, result):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - result[d][i]  # e_j = d_j - y_j
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost

    def test(self, targets):
        """ X: matrix of dimensions n x input """
        green_matrix = self.calc_activation(targets)
        result = dot(green_matrix, self.W)
        return result


def clasification(y):
    if y[0] > 0:
        return 0
    else:
        return 1


def clasification_error(result, targets):
    error = 0.0
    for i in range(len(result)):
        if clasification(result[i]) == clasification(targets[i]):
            error += 1
    pc = error / len(targets)
    return pc


def run(hidden, l=0):
    n = 1000
    x1, y1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 1], [1, 1]], size=n).T
    x2, y2 = np.random.multivariate_normal(mean=[0, 2], cov=[[4, 4], [4, 4]], size=n).T

    dataset = []
    target = []
    for i in range(len(x1)):
        dataset.append([[x1[i]], [y1[i]]])
        target.append([1, -1])

    for i in range(len(x2)):
        dataset.append([[x2[i]], [y2[i]]])
        target.append([-1, 1])

    rbf = RBF(2, hidden, 2)
    rbf.train(dataset, target, lamda=l)
    result = rbf.test(dataset)

    pc = clasification_error(result, target)
    return pc


def run_experiments():
    lambda_values = [0, .1, 1, 10, 100, 1000]
    iterations = 50
    print "\na)"
    hidden = 20
    print "m1: " + str(hidden)
    for l in lambda_values:
        total_error = []
        for i in range(iterations):
            pc = run(10, l)
            total_error.append(pc)
        print "Lambda value: {}".format(l)
        print "\tMean value: {}".format(np.mean(total_error))
        print "\tstandard deviation value: {}".format(np.std(total_error))
        print "\tMax value: {}".format(np.amax(total_error))
        print "\tMin value: {}\n".format(np.amin(total_error))
    print "\nc)"
    hidden = 10
    print "m1: " + str(hidden)
    for l in lambda_values:
        total_error = []
        for i in range(iterations):
            pc = run(10, l)
            total_error.append(pc)
        print "Lambda value: {}".format(l)
        print "\tMean value: {}".format(np.mean(total_error))
        print "\tstandard deviation value: {}".format(np.std(total_error))
        print "\tMax value: {}".format(np.amax(total_error))
        print "\tMin value: {}\n".format(np.amin(total_error))


if __name__ == '__main__':
    run_experiments()
