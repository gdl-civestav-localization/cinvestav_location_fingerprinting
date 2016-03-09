import cPickle
import os
import timeit

import theano.tensor as T
import numpy

from linear_regression import LinearRegression
from mlp import MLP
from utils import load_data


def train(model=None, learning_rate=0.01, l1_learning_rate=0.001, l2_learning_rate=0.0001, n_epochs=1000,
          datasets=None, batch_size=20, name_model='mlp_regressor_mnist.save'):
    """
    Train models.

    :type model: Machine learning model
    :param model: Machine learning model

    :type learning_rate: float
    :param learning_rate: learning rate

    type l1_learning_rate: float
    :param l1_learning_rate: L1-norm's weight when added to the cost

    :type l2_learning_rate: float
    :param l2_learning_rate: L2-norm's weight when added to the cost

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type datasets: Theano shred variable
    :param datasets: Dataset with train, test and valid sets

    :type batch_size: int
    :param batch_size: Size of the batch for train

    :type name_model: String
    :param name_model: Pickle file name
    """
    # compute number of mini batches for training, validation and testing
    n_train_batches = datasets[0][0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = datasets[1][0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches = datasets[2][0].get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    train_model, test_model, validate_model = model.train_functions(
        datasets=datasets,
        batch_size=batch_size,
        l1_learning_rate=l1_learning_rate,
        l2_learning_rate=l2_learning_rate,
        learning_rate=learning_rate
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # Early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 3  # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    done_looping = False

    epoch = 0
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    while (epoch < n_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # print minibatch_avg_cost
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                validation_losses = numpy.mean(validation_losses)

                print 'epoch {}, minibatch {}/{}, validation error {}.'.format(
                    epoch + 1,
                    minibatch_index + 1,
                    n_train_batches,
                    validation_losses * 100.
                )

                # if we got the best validation score until now
                if validation_losses < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if validation_losses < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        print patience

                    # Save best model
                    if epoch > 100:
                        with open(os.path.join('trained_models', name_model), 'wb') as f:
                            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

                    best_validation_loss = validation_losses
                    best_iter = iter

                    # test it on the test set
                    test_score = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_score)

                    print '  -epoch {}, minibatch {}/{}, test error of best model {}.'.format(
                        epoch + 1,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )

            if patience <= iter:
                done_looping = True
                break
        epoch += 1

    end_time = timeit.default_timer()
    print 'Optimization complete. Best validation score of {} obtained at iteration {}, with test performance {}.'.format(
        best_validation_loss * 100.,
        best_iter + 1,
        test_score * 100.
    )
    print 'The code for file {} ran for {}'.format(
        model.__class__.__name__,
        (end_time - start_time) / 60.
    )


def predict(datasets=None, name_model='mlp_regressor_mnist.save'):
    """
    Demonstrates how to test the model

    :type datasets: Theano shred variable
    :param datasets: Dataset with train, test and valid sets
    :type name_model: String
    :param name_model: Pickle file name

    Returns
    -------
    Predictions
    """

    # load the saved model
    with open(os.path.join('trained_models', name_model), 'rb') as f:
        model = cPickle.load(f)

    # We can test it on some examples from test test
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = model.predict(test_set_x)
    desired_values = test_set_y.get_value()
    print "Predicted values test set:"
    for i in range(len(desired_values)):
        print numpy.array(desired_values[i]), ' : ', predicted_values[i]
    return predicted_values


def get_metrics(test_set_y, predicted_values):
    euclidian_error = numpy.array([numpy.linalg.norm(y-d) for d, y in zip(test_set_y, predicted_values)])
    abs_error = numpy.array([numpy.sum(abs(y-d))for d, y in zip(test_set_y, predicted_values)])
    ecm_error = numpy.array([numpy.sum((y-d) ** 2) / len(d) for d, y in zip(test_set_y, predicted_values)])

    def print_metrics(values):
        min_value = values.min()
        max_value = values.max()
        avg_value = values.mean()
        std_value = values.std()
        values.sort()
        print 'Min:', min_value
        print 'Max:', max_value
        print 'Mean:', avg_value
        print 'Std:', std_value
        print '99%:', values[int(len(values) * .99)]
        print '97%:', values[int(len(values) * .97)]
        print '80%:', values[int(len(values) * .80)]
        print '66%:', values[int(len(values) * .66)]
        print '50%:', values[int(len(values) * .50)]
        print '----------------------------------------'
        #print values

    print_metrics(euclidian_error)
    print_metrics(abs_error)
    print_metrics(ecm_error)


if __name__ == '__main__':
    # datasets = load_data('mnist.pkl.gz')

    from datasets import DatasetManager
    datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=True)
    train_set_x, train_set_y = datasets[0]

    n_in = train_set_x.get_value().shape[1]
    print train_set_y.get_value().shape
    n_out = train_set_y.get_value().shape[1]

    x = T.matrix('x')
    linear_regressor_model = LinearRegression(
        input=x,
        n_in=n_in,
        n_out=n_out
    )

    # construct the MLP model
    rng = numpy.random.RandomState(1234)
    mlp_model = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[1000, 1000, 500, 100, 10],
        n_out=n_out,
        activation_function=T.tanh
    )

    model = mlp_model

    train(
        model=model,
        learning_rate=0.001,
        l1_learning_rate=0.001,
        l2_learning_rate=0.0001,
        n_epochs=10000,
        batch_size=600,
        datasets=datasets,
        name_model=model.__class__.__name__ + '_regressor_RSSI20.save'
    )

    # load the saved model
    # with open(os.path.join('trained_models', model.__class__.__name__ + '_regressor_mnist.save'), 'rb') as f:
    #     model = cPickle.load(f)

    predicted_values = predict(
        datasets=datasets,
        name_model=model.__class__.__name__ + '_regressor_RSSI20.save'
    )

    get_metrics(
        test_set_y=datasets[2][1].get_value(),
        predicted_values=predicted_values
    )
