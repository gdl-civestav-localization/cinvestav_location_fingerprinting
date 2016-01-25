__author__ = 'Gibran'

import networkx as nx
import math
import random

random.seed(10)


def rand(a, b):
    return (b - a) * random.random() + a


class NN:
    eta = .1
    total_layers = 0

    def __init__(self, layers, eta=0.1, a=1, b=1, momentum_costant=.1):
        # Activations function values
        self.a = a
        self.b = b
        self.momentum_costant = momentum_costant

        # Layers
        self.total_layers = len(layers)
        self.layers = []

        # Learning rate
        self.eta = eta

        # Add Bias
        layers[0] += 1  # +1 for bias node

        # Create neuronal network
        self.nn = nx.DiGraph()

        # Add nodes
        i = 0
        for m in layers:
            self.layers.append((m, i, m + i))  # Length, Start Index, End index
            for x in xrange(m):
                self.nn.add_node(i, y=0.0, v=0.0, delta=0.0)
                i += 1

        # Add edges
        for l in xrange(len(self.layers) - 1):
            for i in range(self.layers[l][1], self.layers[l][2]):
                for j in range(self.layers[l + 1][1], self.layers[l + 1][2]):
                    self.nn.add_edge(i, j, w=rand(-1.0, 1.0), momentum=0.0)

    def update(self, inputs):
        if len(inputs) != self.layers[0][0] - 1:
            raise ValueError('wrong number of inputs')

        # Set inputs
        for i in range(1, self.layers[0][2]):
            self.nn.node[i]['y'] = inputs[i - 1]
        self.nn.node[0]['y'] = 1  # Set bias

        # Generate output
        for l in range(1, len(self.layers)):  # Iterate for every hidden and output layers
            for j in range(self.layers[l][1], self.layers[l][2]):  # Iterate for every node in actual layer
                v = 0.0
                for i in range(self.layers[l - 1][1], self.layers[l - 1][2]):  # Iterate for every node in past layer
                    v = v + self.nn.node[i]['y'] * self.nn.edge[i][j]['w']
                self.nn.node[j]['v'] = v  # v_j = y_i * w_ji
                self.nn.node[j]['y'] = self.sigmoid(v)  # y_j = phi(v_j)

        # Return network output
        return self.get_output()

    def backpropagate(self, d):  # Desired output
        output_layer = self.total_layers - 1
        if len(d) != self.layers[output_layer][0]:
            raise ValueError('wrong number of target values')

        # Calculate delta for output layer
        for o in range(self.layers[output_layer][1], self.layers[output_layer][2]):  # Iterate for every node in output layer
            index = o - self.layers[output_layer][1]
            error = d[index] - self.nn.node[o]['y']  # e_j = d_j - y_j
            d_phi = self.dsigmoid(self.nn.node[o]['y'])  # Derivative of phi
            self.nn.node[o]['delta'] = error * d_phi

        # Calculate error for hidden layers
        for l in range(len(self.layers) - 2, 0, -1):  # Iterate for every hidden backward
            for j in range(self.layers[l][1], self.layers[l][2]):  # Iterate for every node in actual layer
                d_phi = self.dsigmoid(self.nn.node[j]['y'])  # Derivative of phi

                delta_sum = 0
                for k in range(self.layers[l + 1][1], self.layers[l + 1][2]):  # Iterate for every node in next layer
                    delta_sum += self.nn.node[k]['delta'] * self.nn.edge[j][k]['w']

                self.nn.node[j]['delta'] = d_phi * delta_sum

        # Update weights for output layers
        for o in range(self.layers[output_layer][1], self.layers[output_layer][2]):  # Iterate for every node in output layer
            for j in range(self.layers[output_layer - 1][1], self.layers[output_layer - 1][2]):  # Iterate for every node in past layer
                delta_w = self.get_learning_rate() * self.nn.node[o]['delta'] * self.nn.node[j]['y'] + self.momentum_costant * self.nn.edge[j][o]['momentum']
                self.nn.edge[j][o]['momentum'] = delta_w  # Update momentum
                self.nn.edge[j][o]['w'] += delta_w  # Update weights

        # Update weights for hidden layers
        for l in range(len(self.layers) - 2, 0, -1):  # Iterate for every hidden backward
            for j in range(self.layers[l][1], self.layers[l][2]):  # Iterate for every node in actual layer
                for i in range(self.layers[l - 1][1], self.layers[l - 1][2]):  # Iterate for every node in past layer
                    delta_w = self.get_learning_rate() * self.nn.node[j]['delta'] * self.nn.node[i]['y'] + self.momentum_costant * self.nn.edge[i][j]['momentum']
                    self.nn.edge[i][j]['momentum'] = delta_w  # Update momentum
                    self.nn.edge[i][j]['w'] += delta_w  # Update weights

        # Return error
        return self.cost_function(d)

    def test(self, dataset, targets):
        testpattern = []
        for i in xrange(len(dataset)):
            y = self.update(dataset[i])
            testpattern.append([targets[i], y])
        return testpattern

    def train(self, dataset, targets, theta=.01, iteration=1000):
        print('In progress ')

        errors = []
        i = 0
        while True:
            i += 1
            totalerror = 0
            for j in xrange(len(dataset)):
                d = [targets[j]]
                x = dataset[j]

                self.update(x)
                totalerror = totalerror + self.backpropagate(d)

            totalerror /= len(dataset)
            errors.append(totalerror)
            print "Iteration: " + str(i) + " Error: " + str(totalerror)
            if totalerror <= theta or i >= iteration:
                print "Iterations: " + str(i) + ", Error: " + str(totalerror)
                break
        return errors

    def get_output(self):
        y = []

        output_layer = self.total_layers - 1
        for o in range(self.layers[output_layer][1], self.layers[output_layer][2]):
            y.append(self.nn.node[o]['y'])  # y_j = o_j
        return y

    # Activation function
    def sigmoid(self, v):
        # return 1 / (1 + math.exp(-self.a * v))
        return self.a * math.tanh(self.b * v)

    # derivative of our sigmoid function
    def dsigmoid(self, y):
        # return self.sigmoid(y) * (1 - self.sigmoid(y))
        return (self.b * (self.a - y) * (self.a + y)) / self.a

    def cost_function(self, targets):
        cost = 0.0
        output_layer = self.total_layers - 1
        for o in range(self.layers[output_layer][1], self.layers[output_layer][2]):
            index = o - self.layers[output_layer][1]
            error = abs(targets[index] - self.nn.node[o]['y'])  # e_j = d_j - y_j
            cost += .5 * (error ** 2)
        return cost

    def get_learning_rate(self):
        return self.eta
        # return abs(np.random.normal(self.eta, .1, 1))[0]

