from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
import NeuronalNetwork
import numpy as np

import theano.tensor as T
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne import nonlinearities

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


class DeepBeliefNetwork:
    machine = []
    result = []

    def multilabel_objective(predictions, targets):
        epsilon = np.float32(1.0e-6)
        one = np.float32(1.0)
        pred = T.clip(predictions, epsilon, one - epsilon)
        return -T.sum(targets * T.log(pred) + (one - targets) * T.log(one - pred), axis=1)

    def __init__(self, input_layer, num_output=1):
        self.net = NeuralNet(
                # customize "layers" to represent the architecture you want
                layers=[(layers.InputLayer, {"name": 'input', 'shape': (100, input_layer)}),
                        (layers.DenseLayer, {"name": 'hidden1', 'num_units': 20}),
                        (layers.DenseLayer, {"name": 'output', 'nonlinearity': nonlinearities.sigmoid, 'num_units': 2})], #because you have 13 outputs

                # optimization method:
                update=nesterov_momentum,
                update_learning_rate=5*10**(-3),
                update_momentum=0.9,

                max_epochs=20,  # we want to train this many epochs
                verbose=1,

                #Here are the important parameters for multi labels
                regression=True,

                objective_loss_function=self.multilabel_objective,
                custom_score=("validation score", lambda x, y: np.mean(np.abs(x - y)))
                )

    def train(self, x, y):
        self.net.fit(x, y)

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
