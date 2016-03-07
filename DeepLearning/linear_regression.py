import copy
import numpy
import theano
import theano.tensor as T

__docformat__ = 'restructedtext en'


class LinearRegression(object):
    """Linear regression

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the linear regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.n_in = n_in
        self.n_out = n_out
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        W_values = W
        if W is None:
            W_values = numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        b_values = b
        if b is None:
            b_values = numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # keep track of model input
        self.input = input

        # Output of the model
        self.output = T.dot(self.input, self.W) + self.b  # Linear regression.

        # parameters of the model
        self.params = [self.W, self.b]
        self.L1 = 0
        self.L2 = 0
        for p in self.params:
            self.L1 += T.sum(abs(p))
            self.L2 += T.sum(p ** 2)

    def __getstate__(self):
        print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['params']
        del state['input']
        del state['output']
        del state['L1']
        del state['L2']
        state['W'] = state['W'].get_value()
        state['b'] = state['b'].get_value()
        return state

    def __setstate__(self, state):
        print 'Serializing ' + self.__class__.__name__
        model = LinearRegression(
            input=T.matrix('x'),
            n_in=state['n_in'],
            n_out=state['n_out'],
            W=state['W'],
            b=state['b']
        )
        self.__dict__ = model.__dict__

    def cost(self, y):
        """Return a cost function of the model

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for the output
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        return T.mean((self.output - y) ** 2)

    def predict(self, input):
        """
        Predict function of the model.

        Parameters
        ----------
        input: Matrix of vectors
        """

        # compile a predictor function
        predict_function = theano.function(
            inputs=[self.input],
            outputs=self.output)

        predicted_values = predict_function(input)

        return predicted_values

    def train_functions(self, datasets, batch_size, l1_learning_rate, l2_learning_rate, learning_rate):
        """
        Return a train functions

        :type datasets: Theano shred variable
        :param datasets: Dataset with train, test and valid sets

        :type batch_size: int
        :param batch_size: Size of the batch for train

        type l1_learning_rate: float
        :param l1_learning_rate: L1-norm's weight when added to the cost

        :type l2_learning_rate: float
        :param l2_learning_rate: L2-norm's weight when added to the cost

        :type learning_rate: float
        :param learning_rate: learning rate
        """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        y = T.matrix('y')
        index = T.lscalar()

        # compiling a Theano function that computes the mistakes that are made by the model on a mini batch
        test_model = theano.function(
            inputs=[index],
            outputs=self.cost(y),
            givens={
                self.input: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
        validate_model = theano.function(
            inputs=[index],
            outputs=self.cost(y),
            givens={
                self.input: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        # the cost we minimize during training is the model cost of plus the regularization terms (L1 and L2)
        cost = (
            self.cost(y) + l1_learning_rate * self.L1 + l2_learning_rate * self.L2
        )
        # compute the gradient of cost with respect params
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                self.input: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        return train_model, test_model, validate_model
