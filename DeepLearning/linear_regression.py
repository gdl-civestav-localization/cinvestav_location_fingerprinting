import cPickle
import copy
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from utils import load_data

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
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W_values = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
            W = W_values

        self.W = W

        # initialize the biases b as a vector of n_out 0s
        if b is None:
            b_values = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
            b = b_values

        self.b = b

        # keep track of model input
        self.input = input

        # W is a matrix where column-k represent the separation hyperplane for class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyperplane-k
        self.y_pred = T.dot(self.input, self.W) + self.b  # Linear regression.

        # parameters of the model
        self.params = [self.W, self.b]
        self.L1 = T.sum(abs(self.W)) + T.sum(abs(self.b))
        self.L2 = T.sum(self.W ** 2) + T.sum(self.b ** 2)

    def __getstate__(self):
        print 'Serializing Logistic Regresor'
        state = copy.deepcopy(self.__dict__)
        del state['params']
        del state['input']
        del state['y_pred']
        state['W'] = state['W'].get_value()
        state['b'] = state['b'].get_value()
        return state

    def __setstate__(self, state):
        print 'De-serializing Logistic Regresor'
        self.W = theano.shared(value=state['W'], name='W', borrow=True)
        self.b = theano.shared(value=state['b'], name='b', borrow=True)
        self.input = T.matrix('input')
        self.y_pred = T.dot(self.input, self.W) + self.b
        self.params = [self.W, self.b]

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        return T.mean((self.y_pred - y) ** 2)

    def predict(self, input):
        """
        An example of how to load a trained model and use it
        to predict labels.

        Parameters
        ----------
        input: Matrix of vectors
        """

        # compile a predictor function
        predict_function = theano.function(
            inputs=[self.input],
            outputs=self.y_pred)

        predicted_values = predict_function(input)

        return predicted_values


def train(learning_rate=0.13,
          l1_rate=.001,
          l2_rate=.001,
          n_epochs=1000,
          dataset='mnist.pkl.gz',
          batch_size=600,
          name_model='linear_regresor_mnist.save'):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :param batch_size:

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix(name='x', dtype=theano.config.floatX)  # data, presented as rasterized images
    y = T.matrix(name='y', dtype=theano.config.floatX)  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    regressor = LinearRegression(input=x, n_in=28 * 28, n_out=1)

    # the cost we minimize during training is the root square error

    cost = regressor.errors(y) + l1_rate * regressor.L1 + l2_rate * regressor.L2

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=regressor.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=regressor.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=regressor.W)
    g_b = T.grad(cost=cost, wrt=regressor.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(regressor.W, regressor.W - learning_rate * g_W),
               (regressor.b, regressor.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # mini batch before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print minibatch_avg_cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                """
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )"""

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    # save the best model
                    with open(os.path.join('trained_models', name_model), 'wb') as f:
                        cPickle.dump(regressor, f, protocol=cPickle.HIGHEST_PROTOCOL)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % (end_time - start_time))


def predict(dataset='mnist.pkl.gz', name_model='linear_regresor_mnist.save'):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    with open(os.path.join('trained_models', name_model), 'rb') as f:
        regressor = cPickle.load(f)

    # We can test it on some examples from test test
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = regressor.predict(test_set_x[:100])
    print ("Predicted values for the first 10 examples in test set:")
    print test_set_y.eval()[:100]
    print predicted_values


if __name__ == '__main__':
    train(
        learning_rate=0.01,
        l1_rate=.001,
        l2_rate=.001,
        n_epochs=1000,
        dataset='mnist.pkl.gz',
        batch_size=600,
        name_model='linear_regresor_mnist.save')
    predict(
        dataset='mnist.pkl.gz',
        name_model='linear_regresor_mnist.save')
