"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
import copy
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression
from linear_regression import LinearRegression

__docformat__ = 'restructedtext en'


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation_function=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation_function: theano.Op or function
        :param activation_function: Non linearity to be applied in the hidden layer
        """
        self.input = input

        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid compared to tanh
        #        We have no info for other function, so we use the same as tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation_function == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation_function is None
            else activation_function(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial deep neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (LogisticRegression) or linear regression layer.
    """

    def __init__(self, rng, input, n_in=784, hidden_layers_sizes=[500, 500], n_out=1, activation_function=T.tanh, params=None):
        """Initialize the parameters for the deep multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture

        :type n_in: int
        :param n_in: number of input units

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain at least one value

        :type n_out: int
        :param n_out: number of output units

        :type activation_function: theano.Op or function
        :param activation_function: Non linearity to be applied in the hidden layer

        :type params: List of numpy array
        :param params: free params of the models
        """

        self.n_in = n_in
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_out = n_out
        self.activation_function = activation_function

        self.hidden_layers = []
        self.params = []
        self.n_layers = len(self.hidden_layers_sizes)

        # keep track of model input
        self.input = input

        assert self.n_layers > 0

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = self.n_in
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.hidden_layers[-1].output

            # Set params W and b from params for hidden layer
            W_val = None
            b_val = None
            if params is not None:
                W_val = params[i * 2]
                b_val = params[i * 2 + 1]
            hiddenLayer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=input_size,
                n_out=self.hidden_layers_sizes[i],
                activation_function=activation_function,
                W=W_val,
                b=b_val
            )

            # add the layer to our list of layers
            self.hidden_layers.append(hiddenLayer)

            # add parameter of hidden layer to params
            self.params.extend(hiddenLayer.params)

        # We now need to add top of the MLP
        W_val = None
        b_val = None
        if params is not None:
            W_val = params[-2]
            b_val = params[-1]
        self.outputLayer = LinearRegression(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_out,
            W=W_val,
            b=b_val
        )
        self.params.extend(self.outputLayer.params)

        # Output of the model
        self.output = self.outputLayer.output

        self.L1 = 0
        self.L2 = 0
        for p in self.params:
            # L1 norm one regularization option is to enforce L1 norm to be small
            self.L1 += T.sum(abs(p))
            # square of L2 norm one regularization option is to enforce square of L2 norm to be small
            self.L2 += T.sum(p ** 2)

    def __getstate__(self):
        print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['output']
        del state['input']
        del state['L2']
        del state['L1']
        del state['hidden_layers']
        del state['outputLayer']
        if state['activation_function'] == theano.tensor.nnet.sigmoid:
            state['activation_function'] == 'sigmoid'
        else:
            state['activation_function'] == 'tanh'
        for i, val in enumerate(state['params']):
            state['params'][i] = val.get_value(borrow=True)
        return state

    def __setstate__(self, state):
        print 'De-serializing ' + self.__class__.__name__
        if state['activation_function'] == 'sigmoid':
            activation_function = theano.tensor.nnet.sigmoid
        else:
            activation_function = T.tanh

        mlp = MLP(
            rng=numpy.random.RandomState(),
            input=T.matrix('x'),
            n_in=state['n_in'],
            hidden_layers_sizes=state['hidden_layers_sizes'],
            n_out=state['n_out'],
            params=state['params'],
            activation_function=activation_function
        )
        self.__dict__ = mlp.__dict__

    def cost(self, y):
        """Return a cost function of the model

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for the output
        """
        if y.ndim != self.output.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type)
            )
        return T.mean((self.output - y) ** 2)
        # return T.sum((self.output - y) ** 2)

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
            outputs=self.cost(y),
            updates=updates,
            givens={
                self.input: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        return train_model, test_model, validate_model
