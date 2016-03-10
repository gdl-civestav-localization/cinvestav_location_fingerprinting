import cPickle
import os
import timeit

import theano
import theano.tensor as T
import numpy

from DBN import DBN
from linear_regression import LinearRegression
from mlp import MLP
from utils import load_data


def train_functions(model, datasets, batch_size, l1_learning_rate, l2_learning_rate, learning_rate):
        """
        Generates a function `train` that implements one step of fine-tuning,
        a function `validate` that computes the error on a batch from the validation set
        and a function `test` that computes the error on a batch from the testing set

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
            outputs=model.cost(y),
            givens={
                model.input: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            inputs=[index],
            outputs=model.cost(y),
            givens={
                model.input: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # the cost we minimize during training is the model cost of plus the regularization terms (L1 and L2)
        cost = (
            model.cost(y) + l1_learning_rate * model.L1 + l2_learning_rate * model.L2
        )
        # compute the gradient of cost with respect params
        gparams = [T.grad(cost, param) for param in model.params]

        # compute list of fine-tuning updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(model.params, gparams)
            ]
        train_model = theano.function(
            inputs=[index],
            outputs=model.cost(y),
            updates=updates,
            givens={
                model.input: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        return train_model, test_model, validate_model


def train(
        model=None,
        learning_rate=0.01,
        l1_learning_rate=0.001,
        l2_learning_rate=0.0001,
        n_epochs=1000,
        datasets=None,
        batch_size=20,
        name_model='mlp_regressor_mnist.save',
        pre_training_epochs=10,
        pre_train_lr=0.01,
        k=1
):
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

    :type pre_training_epochs: int
    :param pre_training_epochs: number of epoch to do pre-training

    :type pre_train_lr: float
    :param pre_train_lr: learning rate to be used during pre-training

    :type k: int
    :param k: number of Gibbs steps to do in CD-k / PCD-k
    """
    # compute number of mini batches for training, validation and testing
    n_train_batches = datasets[0][0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = datasets[1][0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches = datasets[2][0].get_value(borrow=True).shape[0] / batch_size

    #########################
    # PRETRAINING THE MODEL #
    #########################
    if hasattr(model, 'pre_training_functions'):
        pre_train_model(
            model=model,
            datasets=datasets,
            batch_size=batch_size,
            pre_training_epochs=pre_training_epochs,
            pre_train_lr=pre_train_lr,
            k=k
        )

    ###############
    # TRAIN MODEL #
    ###############
    train_model, test_model, validate_model = train_functions(
        model=model,
        datasets=datasets,
        batch_size=batch_size,
        l1_learning_rate=l1_learning_rate,
        l2_learning_rate=l2_learning_rate,
        learning_rate=learning_rate
    )

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

            mini_batch_avg_cost = train_model(minibatch_index)
            # print mini_batch_avg_cost

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


def pre_train_model(model, datasets=None, batch_size=20, pre_training_epochs=10, pre_train_lr=0.01, k=1):
    """
    Train models.

    :type model: Machine learning model
    :param model: Machine learning model

    :type datasets: Theano shred variable
    :param datasets: Dataset with train, test and valid sets

    :type batch_size: int
    :param batch_size: Size of the batch for train

    :type pre_training_epochs: int
    :param pre_training_epochs: number of epoch to do pre-training

    :type pre_train_lr: float
    :param pre_train_lr: learning rate to be used during pre-training

    :type k: int
    :param k: number of Gibbs steps to do in CD-k / PCD-k
    """

    print '... getting the pre-training functions'
    n_train_batches = datasets[0][0].get_value(borrow=True).shape[0] / batch_size

    pre_training_fns = model.pre_training_functions(
        datasets=datasets,
        batch_size=batch_size,
        k=k
    )

    print '... pre-training the model'
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in xrange(model.n_layers):
        for epoch in xrange(pre_training_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pre_training_fns[i](
                    index=batch_index,
                    lr=pre_train_lr
                ))
            print 'Pre-training layer {}, epoch {}, cost {}'.format(
                i,
                epoch,
                numpy.mean(c)
            )

    end_time = timeit.default_timer()
    print 'The pre-training code for file {} ran for {}m'.format(
        os.path.split(__file__)[1],
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
        # print values

    print_metrics(euclidian_error)
    print_metrics(abs_error)
    print_metrics(ecm_error)


if __name__ == '__main__':
    # datasets = load_data('mnist.pkl.gz')

    from datasets import DatasetManager
    datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=True, seed=20)
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

    dbn_model = DBN(
        numpy_rng=rng,
        n_visible=n_in,
        theano_rng=None,
        hidden_layers_sizes=[1000, 1000, 500, 100, 10],
        n_outs=n_out
    )

    gbrbm_dbn_model = DBN(
        numpy_rng=rng,
        n_visible=n_in,
        theano_rng=None,
        hidden_layers_sizes=[1000, 1000, 500, 100, 10],
        n_outs=n_out,
        gaussian_visible=True
    )

    model = linear_regressor_model

    train(
        model=model,
        learning_rate=0.001,
        l1_learning_rate=0.001,
        l2_learning_rate=0.0001,
        n_epochs=10000,
        batch_size=600,
        datasets=datasets,
        name_model=model.__class__.__name__ + '_regressor_RSSI20.save',
        pre_training_epochs=10,
        pre_train_lr=0.001,
        k=1
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
