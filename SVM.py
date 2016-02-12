from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
import NeuronalNetwork
import numpy as np

__author__ = 'Usuario'


class SVM:
    machine = []
    result = []

    def __init__(self, num_output=1):
        for i in xrange(num_output):
            clf = svm.SVR(kernel='rbf', C=1e2, gamma=8, epsilon=.1)
            self.machine.append(clf)

    def train(self, x, y):
        result = zip(*y)
        for i in xrange(len(self.machine)):
            self.machine[i].fit(x, result[i])

    def test(self, dataset, targets):
        self.result = []
        for j in xrange(len(dataset)):
            y = []
            for i in xrange(len(self.machine)):
                y.append(self.machine[i].predict(dataset[j])[0])
            self.result.append(y)
        return self.result, self.cost_function(targets)

    def cost_function(self, targets):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - self.result[d][i]  # e_j = d_j - y_j
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost


class AdaBoosting:
    machine = []
    result = []

    def __init__(self, num_output=1):
        for i in xrange(num_output):
            clf = GradientBoostingRegressor(n_estimators=100, learning_rate=.1, max_depth=5, loss='ls')
            self.machine.append(clf)

    def train(self, x, y):
        result = zip(*y)
        for i in xrange(len(self.machine)):
            self.machine[i].fit(x, result[i])

    def test(self, dataset, targets):
        self.result = []
        for j in xrange(len(dataset)):
            y = []
            for i in xrange(len(self.machine)):
                y.append(self.machine[i].predict(dataset[j])[0])
            self.result.append(y)
        return self.result, self.cost_function(targets)

    def cost_function(self, targets):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - self.result[d][i]  # e_j = d_j - y_j
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost


class NeuronalNetworks:
    machine = []
    result = []

    def __init__(self, input_layer, num_output=1):
        for i in xrange(num_output):
            clf = NeuronalNetwork.NN(np.array([input_layer, 30, 1]), eta=.1, a=40, b=.001, momentum_costant=.8)
            self.machine.append(clf)

    def train(self, dataset, targets):
        result = zip(*targets)
        errors = []
        for i in xrange(len(self.machine)):
            train_error = self.machine[i].train(dataset=dataset, targets=result[i], theta=2.85, iteration=1000)
            errors.append(train_error)
        return errors

    def test(self, dataset, targets):
        self.result = []
        for j in xrange(len(dataset)):
            y = []
            for i in xrange(len(self.machine)):
                ss = self.machine[i].test(dataset=[dataset[j]], targets=[targets[j]])
                y.append(ss[0][1])
            self.result.append(y)
        return self.result, self.cost_function(targets)

    def cost_function(self, targets):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - self.result[d][i]  # e_j = d_j - y_j
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost
