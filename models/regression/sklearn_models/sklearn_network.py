import numpy as np
from copy import deepcopy


class SklearnNetwork:
    def __init__(self, sklearn_model, num_output=1):
        self.machine = []
        self.result = []
        for i in xrange(num_output):
            clf = sklearn_model
            self.machine.append(deepcopy(clf))

    def train(self, x, y):
        result = zip(*y)
        for i in xrange(len(self.machine)):
            self.machine[i].fit(x, result[i])

    def test(self, dataset):
        self.result = []
        y = []
        for i in xrange(len(self.machine)):
            y.append(self.machine[i].predict(dataset))

        self.result = np.array(y).T
        return np.array(self.result)

    def cost_function(self, targets):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - self.result[d][i]  # e_j = d_j - y_j
                # cost += abs(error)
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost
