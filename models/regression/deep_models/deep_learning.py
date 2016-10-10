import cPickle
import os
import timeit

import sys
import theano
import theano.tensor as T
import numpy


def train_functions(model, datasets, batch_size, learning_rate, annealing_learning_rate,
                    l1_learning_rate, l2_learning_rate):
    """
        Generates a function `train` that implements one step of fine-tuning,
        a function `validate` that computes the error on a batch from the validation set
        and a function `test` that computes the error on a batch from the testing set

        :type datasets: Theano shred variable
        :param datasets: Dataset with train, test and valid sets

        :type batch_size: int
        :param batch_size: Size of the batch for train

        :type learning_rate: float
        :param learning_rate: learning rate

        :type annealing_learning_rate: float
        :param annealing_learning_rate: decreasing rate of learning rate

        type l1_learning_rate: float
        :param l1_learning_rate: L1-norm's weight when added to the cost

        :type l2_learning_rate: float
        :param l2_learning_rate: L2-norm's weight when added to the cost
        """
    train_set_x, train_set_y = datasets['train_set']
    valid_set_x, valid_set_y = datasets['valid_set']
    test_set_x, test_set_y = datasets['test_set']

    y = T.matrix('y')
    index = T.lscalar()

    # compiling a Theano function that computes the mistakes that are made by the model on a mini batch
    test_model = theano.function(
        inputs=[index],
        outputs=error_function(model, y),
        givens={
            model.input: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=error_function(model, y),
        givens={
            model.input: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the cost we minimize during training is the model cost of plus the regularization terms (L1 and L2)
    loss_function = (
        cost_function(model, y) + l1_learning_rate * model.L1 + l2_learning_rate * model.L2
    )

    # compute the gradient of cost with respect params
    gparams = [T.grad(loss_function, param) for param in model.params]

    #################################################
    # Wudi change the annealing learning rate:
    #################################################
    updates = []
    state_learning_rate = theano.shared(
        numpy.asarray(
            learning_rate,
            dtype=theano.config.floatX
        ),
        borrow=True)
    updates.append((state_learning_rate, annealing_learning_rate * state_learning_rate))

    # compute list of fine-tuning updates
    for param, gparam in zip(model.params, gparams):
        updates.append((param, param - state_learning_rate * gparam))

    train_model = theano.function(
        inputs=[index],
        outputs=loss_function,
        updates=updates,
        givens={
            model.input: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    return train_model, test_model, validate_model


def cost_function(model, y):
    """Return a cost function of the model

        :type model: Machine learning model
        :param model: Machine learning model

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for the output
        """
    if y.ndim != model.output.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', model.output.type)
        )
    return T.mean(.5 * (model.output - y) ** 2)


def error_function(model, y):
    """Return a cost function of the model

        :type model: Machine learning model
        :param model: Machine learning model

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for the output
        """
    if y.ndim != model.output.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', model.output.type)
        )
    return T.mean(.5 * (model.output - y) ** 2)
    # return T.mean(T.abs_(model.output - y))


def train(
        model,
        name_model,
        datasets=None,
        learning_rate=0.01,
        annealing_learning_rate=.999,
        l1_learning_rate=0.001,
        l2_learning_rate=0.0001,
        n_epochs=1000,
        batch_size=20,
        pre_training_epochs=10,
        pre_train_lr=0.01,
        k=1
):
    """
    Train sklearn_models.

    :type model: Machine learning model
    :param model: Machine learning model

    :type learning_rate: float
    :param learning_rate: learning rate

    :type annealing_learning_rate: float
    :param annealing_learning_rate: decreasing rate of learning rate

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
    n_train_batches = datasets['train_set'][0].get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = datasets['valid_set'][0].get_value(borrow=True).shape[0] / batch_size
    n_test_batches = datasets['test_set'][0].get_value(borrow=True).shape[0] / batch_size

    print '---------------------------------------', model.__class__.__name__, '---------------------------------------'

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
        learning_rate=learning_rate,
        annealing_learning_rate=annealing_learning_rate,
        l1_learning_rate=l1_learning_rate,
        l2_learning_rate=l2_learning_rate
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
    best_epoch = 0
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
                validation_losses = numpy.mean(validation_losses) * 100.

                if 'pydevd' in sys.modules:
                    print 'epoch {}, minibatch {}/{}, validation error {}.'.format(
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        validation_losses
                    )

                # if we got the best validation score until now
                if validation_losses < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if validation_losses < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        # print patience

                    # Save best model
                    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'trained_models', name_model), 'wb') as f:
                        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

                    best_validation_loss = validation_losses
                    best_epoch = epoch

                    # test it on the test set
                    test_score = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_score) * 100.

                    if 'pydevd' in sys.modules:
                        print '  -epoch {}, minibatch {}/{}, test error of best model {}.'.format(
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score
                        )

            # if patience <= iter:
                # done_looping = True
                # break
        epoch += 1

    end_time = timeit.default_timer()
    print 'Optimization complete. \nBest validation score {} obtained at epoch {}, \n' \
        'Test performance {}.\nIteration {}'.format(
            best_validation_loss,
            best_epoch,
            test_score,
            epoch
        )

    print 'Model: {}, ran for {}'.format(
        model.__class__.__name__,
        (end_time - start_time)
    )

    return best_validation_loss, test_score


def pre_train_model(model, datasets=None, batch_size=20, pre_training_epochs=10, pre_train_lr=0.01, k=1):
    """
    Train sklearn_models.

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
    n_train_batches = datasets['dataset_unlabeled'].get_value(borrow=True).shape[0] / batch_size

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
            # if 'pydevd' in sys.modules:
            print 'Pre-training layer {}, epoch {}, cost {}'.format(
                i,
                epoch,
                numpy.mean(c)
            )

    end_time = timeit.default_timer()
    print 'The pre-training code ran for {}m'.format(
        (end_time - start_time)
    )


def predict(prediction_set, name_model):
    """
    Demonstrates how to test the model

    :type prediction_set: Theano shred variable
    :param prediction_set: Input values to make a prediction
    :type name_model: String
    :param name_model: Pickle file name

    Returns
    -------
    Predictions
    """

    # load the saved model
    with open(os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'trained_models', name_model), 'rb') as f:
        model = cPickle.load(f)

    predicted_values = model.predict(prediction_set)
    return predicted_values
