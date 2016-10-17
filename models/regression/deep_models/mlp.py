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
import sys
import theano
import theano.tensor as T
from theano.tensor.nnet import batch_normalization
from linear_regression import LinearRegression
from theano.tensor.shared_randomstreams import RandomStreams

__docformat__ = 'restructedtext en'


class HiddenLayer(object):
    def __init__(
            self,
            rng,
            input,
            n_in,
            n_out,
            W=None,
            b=None,
            gamma=None,
            beta=None,
            activation_function=T.tanh
    ):
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
        W_values = W
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
        if isinstance(W_values, numpy.ndarray):
            W_values = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W_values

        b_values = b
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        if isinstance(b_values, numpy.ndarray):
            b_values = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b_values

        gamma_val = gamma
        if gamma is None:
            gamma_val = numpy.ones((n_out,), dtype=theano.config.floatX)
        if isinstance(gamma_val, numpy.ndarray):
            gamma_val = theano.shared(value=gamma_val, name='gamma', borrow=True)
        self.gamma = gamma_val

        beta_val = beta
        if beta is None:
            beta_val = numpy.zeros((n_out,), dtype=theano.config.floatX)
        if isinstance(beta_val, numpy.ndarray):
            beta_val = theano.shared(value=beta_val, name='beta', borrow=True)
        self.beta = beta_val

        # linear output
        lin_output = T.dot(input, self.W) + self.b

        bn_output = batch_normalization(
            inputs=lin_output,
            gamma=self.gamma,
            beta=self.beta,
            mean=lin_output.mean((0,), keepdims=True),
            std=lin_output.std((0,), keepdims=True),
            mode='high_mem'
        )

        if activation_function is None:
            self.output = bn_output
        elif activation_function == T.nnet.relu:
            self.output = T.clip(bn_output, 0, 20)
        else:
            self.output = activation_function(bn_output)

        # parameters of the model
        self.params = [self.W, self.b, self.gamma, self.beta]


class DropoutHiddenLayer(HiddenLayer):
    def __init__(
            self,
            rng,
            input,
            n_in,
            n_out,
            dropout_rate,
            W=None,
            b=None,
            gamma=None,
            beta=None,
            activation_function=T.tanh
    ):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_out,
            W=W,
            b=b,
            gamma=gamma,
            beta=beta,
            activation_function=activation_function
        )
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        mask = theano_rng.binomial(
            n=1,
            p=1 - dropout_rate,
            size=self.output.shape,
            dtype=theano.config.floatX
        )
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        dropout_output = self.output * T.cast(mask, theano.config.floatX)

        self.output = dropout_output


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial deep neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (LogisticRegression) or linear regression layer.
    """

    def __init__(
            self,
            numpy_rng,
            input,
            n_in=784,
            hidden_layers_sizes=None,
            n_out=1,
            dropout_rate=None,
            activation_function=T.tanh,
            params=None
    ):
        """Initialize the parameters for the deep multilayer perceptron

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to initialize weights

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
        :param params: free params of the sklearn_models
        """
        self.n_in = n_in
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_out = n_out
        self.activation_function = activation_function

        self.hidden_layers = []
        self.drop_hidden_layers = []
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
                drop_layer_input = self.input
            else:
                layer_input = self.hidden_layers[-1].output
                if dropout_rate is not None:
                    drop_layer_input = self.drop_hidden_layers[-1].output

            # Set params W and b from params for hidden layer
            W_val = None
            b_val = None
            gamma_val = None
            beta_val = None
            if params is not None:
                W_val = params[i * 4]
                b_val = params[i * 4 + 1]
                gamma_val = params[i * 4 + 2]
                beta_val = params[i * 4 + 3]

            hidden_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=self.hidden_layers_sizes[i],
                activation_function=activation_function,
                W=W_val,
                b=b_val,
                gamma=gamma_val,
                beta=beta_val
            )
            self.hidden_layers.append(hidden_layer)

            if dropout_rate is not None:
                drop_hidden_layer = DropoutHiddenLayer(
                    rng=numpy_rng,
                    input=drop_layer_input,
                    n_in=input_size,
                    n_out=self.hidden_layers_sizes[i],
                    activation_function=activation_function,
                    W=hidden_layer.W / (1 - dropout_rate),
                    b=hidden_layer.b,
                    gamma=hidden_layer.gamma,
                    beta=hidden_layer.beta,
                    dropout_rate=dropout_rate
                )
                self.drop_hidden_layers.append(drop_hidden_layer)

            # add parameter of hidden layer to params
            self.params.extend(hidden_layer.params)

        # We now need to add top of the MLP
        W_val = None
        b_val = None
        gamma_val = None
        beta_val = None
        if params is not None:
            W_val = params[-4]
            b_val = params[-3]
            gamma_val = params[-2]
            beta_val = params[-1]
        linear_regression = LinearRegression(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_out,
            W=W_val,
            b=b_val,
            gamma=gamma_val,
            beta=beta_val
        )
        self.outputLayer = linear_regression

        if dropout_rate is not None:
            drop_linear_regression = LinearRegression(
                input=self.drop_hidden_layers[-1].output,
                n_in=self.hidden_layers_sizes[-1],
                n_out=self.n_out,
                W=linear_regression.W / (1 - dropout_rate),
                b=linear_regression.b,
                gamma=linear_regression.gamma,
                beta=linear_regression.beta,
            )
            self.drop_output_layer = drop_linear_regression

        self.params.extend(self.outputLayer.params)

        # Output of the model
        self.output = self.outputLayer.output
        if dropout_rate is not None:
            self.drop_output = self.drop_output_layer.output

        self.L1 = 0
        self.L2 = 0
        for i in range(0, len(self.params), 4):
            p = self.params[i]  # weights
            # L1 norm one regularization option is to enforce L1 norm to be small
            self.L1 += T.sum(abs(p))
            # square of L2 norm one regularization option is to enforce square of L2 norm to be small
            self.L2 += T.sum(p ** 2)

    def __getstate__(self):
        if 'pydevd' in sys.modules:
            print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['input']
        del state['L2']
        del state['L1']
        del state['hidden_layers']
        del state['drop_hidden_layers']
        del state['output']
        del state['drop_output']
        del state['outputLayer']
        del state['drop_output_layer']

        if state['activation_function'] == theano.tensor.nnet.sigmoid:
            state['activation_function'] = 'sigmoid'
        elif state['activation_function'] == T.nnet.relu:
            state['activation_function'] = 'relu'
        else:
            state['activation_function'] = 'tanh'

        for i, val in enumerate(state['params']):
            state['params'][i] = val.get_value(borrow=True)
        return state

    def __setstate__(self, state):
        if 'pydevd' in sys.modules:
            print 'De-serializing ' + self.__class__.__name__
        if state['activation_function'] == 'sigmoid':
            activation_function = theano.tensor.nnet.sigmoid
        elif state['activation_function'] == 'relu':
            activation_function = theano.tensor.nnet.relu
        else:
            activation_function = T.tanh

        mlp = MLP(
            numpy_rng=numpy.random.RandomState(),
            input=T.matrix('x'),
            n_in=state['n_in'],
            hidden_layers_sizes=state['hidden_layers_sizes'],
            n_out=state['n_out'],
            params=state['params'],
            dropout_rate=state['dropout_rate'],
            activation_function=activation_function
        )
        self.__dict__ = mlp.__dict__

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
