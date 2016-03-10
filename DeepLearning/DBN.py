import copy
import numpy
import sys
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from grbm import GBRBM
from linear_regression import LinearRegression
from mlp import HiddenLayer
from rbm import RBM


class DBN(object):
    """
    Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_visible=784,
                 hidden_layers_sizes=None, n_outs=10, params=None, gaussian_visible=False):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_visible: int
        :param n_visible: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type params: List of numpy array
        :param params: free params of the models

        :type gaussian_visible: Boolean
        :param gaussian_visible: True if the visible units are gaussian
        """
        # Params to reconstruct the DBN
        self.n_visible = n_visible
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_outs = n_outs

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(self.hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.input = T.matrix('x')

        # The DBN is an MLP, for which all weights of intermediate layers are shared with a different RBM.
        # We will first construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoid layer we also construct an RBM
        # that shares weights with that layer.
        # During pre-training we
        # will train these RBMs (which will lead to changing the weights of the MLP as well)
        # During fine-tuning we will finish training the DBN by doing stochastic gradient descent on the MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoid layer
            if i == 0:
                input_size = self.n_visible
                layer_input = self.input
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = self.sigmoid_layers[-1].output

            # Set params W and b from params for hidden layer
            W_val = None
            b_val = None
            if params is not None:
                W_val = params[i * 2]
                b_val = params[i * 2 + 1]

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=self.hidden_layers_sizes[i],
                activation_function=T.nnet.sigmoid,
                W=W_val,
                b=b_val
            )

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            if i == 0 and gaussian_visible:
                self.__class__.__name__ = "GBRBM-DBN"
                rbm_layer = GBRBM(
                    numpy_rng=numpy_rng,
                    theano_rng=theano_rng,
                    input=layer_input,
                    n_visible=input_size,
                    n_hidden=self.hidden_layers_sizes[i],
                    W=sigmoid_layer.W,
                    h_bias=sigmoid_layer.b
                )
            else:
                rbm_layer = RBM(
                    numpy_rng=numpy_rng,
                    theano_rng=theano_rng,
                    input=layer_input,
                    n_visible=input_size,
                    n_hidden=self.hidden_layers_sizes[i],
                    W=sigmoid_layer.W,
                    h_bias=sigmoid_layer.b
                )
            self.rbm_layers.append(rbm_layer)

        # We now need to add top of the MLP
        W_val = None
        b_val = None
        if params is not None:
            W_val = params[-2]
            b_val = params[-1]
        self.outputLayer = LinearRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layers_sizes[-1],
            n_out=self.n_outs,
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
        if 'pydevd' in sys.modules:
            print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['input']
        del state['output']
        del state['rbm_layers']
        del state['sigmoid_layers']
        del state['outputLayer']
        del state['L1']
        del state['L2']
        for i, val in enumerate(state['params']):
            state['params'][i] = val.get_value(borrow=True)
        return state

    def __setstate__(self, state):
        if 'pydevd' in sys.modules:
            print 'De-serializing ' + self.__class__.__name__
        dbn = DBN(
            numpy_rng=numpy.random.RandomState(),
            theano_rng=None,
            n_visible=state['n_visible'],
            hidden_layers_sizes=state['hidden_layers_sizes'],
            n_outs=state['n_outs'],
            params=state['params'])
        self.__dict__ = dbn.__dict__

    def cost(self, y):
        """
        Return a cost function of the model

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

    def pre_training_functions(self, datasets, batch_size, k=1):
        """
        Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the mini batch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        mini batch indexes.

        :type datasets: Theano shred variable
        :param datasets: Dataset with train, test and valid sets

        :type batch_size: int
        :param batch_size: size of a mini batch

        :type k: int
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        """
        train_set_x, train_set_y = datasets[0]

        # index to a [mini]batch
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

        # number of batches
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pre_train_fns = []
        for rbm in self.rbm_layers:
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, learning_rate],
                outputs=cost,
                updates=updates,
                givens={
                    self.input: train_set_x[batch_begin:batch_end]
                }
            )
            # append fn to the list of functions
            pre_train_fns.append(fn)

        return pre_train_fns

    def predict(self, input):
        """
        Predict function of the model.

        Parameters
        ----------
        input: Matrix of vectors
        """

        predict_function = theano.function(
            inputs=[self.input],
            outputs=self.output)

        predicted_values = predict_function(input)
        return predicted_values
