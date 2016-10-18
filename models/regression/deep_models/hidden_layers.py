import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import batch_normalization
from theano.tensor.shared_randomstreams import RandomStreams


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
            mean=lin_output.mean(),
            std=lin_output.std(),
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