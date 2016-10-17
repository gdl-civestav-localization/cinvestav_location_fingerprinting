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
from models.regression.deep_models.hidden_layers import HiddenLayer, DropoutHiddenLayer
from theano.tensor.shared_randomstreams import RandomStreams

__docformat__ = 'restructedtext en'


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
        self.n_layers = len(self.hidden_layers_sizes)
        self.n_out = n_out
        self.activation_function = activation_function

        self.hidden_layers = []
        self.drop_hidden_layers = []

        self.dropout_rate = dropout_rate
        self.drop_output = None
        self.drop_output_layer = None

        self.params = []

        # keep track of model input
        self.input = input

        assert self.n_layers > 0

        for i in range(self.n_layers):
            if i == 0:
                input_size = self.n_in
                layer_input = self.input
                drop_layer_input = self.input
            else:
                input_size = self.hidden_layers_sizes[i - 1]
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
        linear_regression = HiddenLayer(
            rng=numpy_rng,
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_out,
            activation_function=None,
            W=W_val,
            b=b_val,
            gamma=gamma_val,
            beta=beta_val
        )
        self.outputLayer = linear_regression

        if dropout_rate is not None:
            drop_linear_regression = HiddenLayer(
                rng=numpy_rng,
                input=self.drop_hidden_layers[-1].output,
                n_in=self.hidden_layers_sizes[-1],
                n_out=self.n_out,
                activation_function=None,
                W=linear_regression.W / (1 - dropout_rate),
                b=linear_regression.b,
                gamma=linear_regression.gamma,
                beta=linear_regression.beta
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
