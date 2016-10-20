import copy
import numpy
import sys
import theano
import theano.tensor as T
from theano.tensor.nnet import batch_normalization


class LinearRegression(object):
    """Linear regression

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None, gamma=None, beta=None):
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
            W_values = numpy.asarray(
                numpy.random.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = b
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        gamma_val = gamma
        if gamma is None:
            gamma_val = numpy.ones((n_out,), dtype=theano.config.floatX)
        self.gamma = theano.shared(value=gamma_val, name='gamma', borrow=True)

        beta_val = beta
        if beta is None:
            beta_val = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.beta = theano.shared(value=beta_val, name='beta', borrow=True)

        # keep track of model input
        self.input = input

        # Linear regression.
        linear = T.dot(self.input, self.W) + self.b
        bn_output = batch_normalization(
            inputs=linear,
            gamma=self.gamma,
            beta=self.beta,
            mean=linear.mean((0,), keepdims=True),
            std=linear.std((0,), keepdims=True),
            mode='high_mem'
        )

        # Output of the model
        self.output = bn_output

        # parameters of the model
        self.params = [self.W, self.b, self.gamma, self.beta]
        self.L1 = T.sum(abs(self.W))
        self.L2 = T.sum(self.W ** 2)

    def __getstate__(self):
        if 'pydevd' in sys.modules:
            print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['params']
        del state['input']
        del state['output']
        del state['L1']
        del state['L2']
        state['W'] = state['W'].get_value()
        state['b'] = state['b'].get_value()
        state['gamma'] = state['gamma'].get_value()
        state['beta'] = state['beta'].get_value()
        return state

    def __setstate__(self, state):
        if 'pydevd' in sys.modules:
            print 'Serializing ' + self.__class__.__name__
        model = LinearRegression(
            input=T.matrix('x'),
            n_in=state['n_in'],
            n_out=state['n_out'],
            W=state['W'],
            b=state['b'],
            gamma=state['gamma'],
            beta=state['beta']
        )
        self.__dict__ = model.__dict__

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
