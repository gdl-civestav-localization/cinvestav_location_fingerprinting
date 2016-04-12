from sklearn import neighbors


class KNN:
    def __init__(self, num_output=1):
        self.machine = []
        self.result = []
        for i in xrange(num_output):
            knn = neighbors.KNeighborsRegressor(3, weights='distance')
            self.machine.append(knn)

    def train(self, x, y):
        result = zip(*y)
        for i in xrange(len(self.machine)):
            self.machine[i].fit(x, result[i])

    def test(self, dataset):
        self.result = []
        for j in xrange(len(dataset)):
            y = []
            for i in xrange(len(self.machine)):
                y.append(self.machine[i].predict(dataset[j])[0])
            self.result.append(y)
        return self.result

    def cost_function(self, targets):
        cost = 0.0
        for d in xrange(len(targets)):
            for i in xrange(len(targets[d])):
                error = targets[d][i] - self.result[d][i]  # e_j = d_j - y_j
                cost += .5 * (error ** 2)
        cost /= len(targets)
        return cost
