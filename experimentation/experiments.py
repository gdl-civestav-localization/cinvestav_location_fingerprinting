import os
import sys
import cPickle
import numpy as np
import pandas as pd


def run_experiments_sklearn(models, seed, params, experiment_name, task_type='regression'):
    datasets = params['datasets']
    params['datasets'] = None

    train_set_x, train_set_y = datasets['train_set']
    test_set_x, test_set_y = datasets['test_set']
    prediction_set_x = datasets['prediction_set']

    results = [('params', params)]
    predictions = []
    for name, model in models:
        print '---------------------------------------', name, '---------------------------------------'

        # Train
        model.train(train_set_x, train_set_y)

        # Test
        print "Test values test set:"
        predicted_values = model.test(test_set_x)
        # Euclidean distance
        test_error = np.mean(np.linalg.norm(test_set_y - predicted_values, axis=1))
        print 'Test error (Euclidean distance): ', test_error

        test_error2 = np.mean(np.abs(test_set_y - predicted_values), axis=0)
        print 'Test mean error (X, Y): ', test_error2

        for i in range(len(test_set_y)):
            print test_set_y[i], ' : ', predicted_values[i]

        # Predict
        predicted_values = model.test(prediction_set_x)
        predictions.append(predicted_values.flatten())

        dict_results = {
            'Name': name,
            'test_score': test_error,
            'test_error_x_y': test_error2,
            'predicted_values': predicted_values,
            'test_set': test_set_y,
            'seed': seed
        }
        results.append(dict_results)

    # Save results
    experiments_path = os.path.join(
        os.path.dirname(sys.argv[0]), 'experiments', experiment_name + '_sklearn_seed_' + str(seed)
    )

    with open(experiments_path, 'wb') as f:
        cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # Save predictions
    df = pd.DataFrame(
        data=np.array(predictions).transpose(),
        columns=[name for name, model in models]
    )
    prediction_name = os.path.join(os.path.dirname(sys.argv[0]), 'predictions', 'sklearn_predictions' + '.csv')
    df.to_csv(
        prediction_name,
        sep=',',
        encoding='utf-8',
        columns=[name for name, model in models]
    )


def run_theano_experiments(models, seed, params, experiment_name, task_type='regression'):
    if task_type == 'regression':
        from models.regression.deep_models.deep_learning import train, predict
    else:
        raise NotImplementedError()

    datasets = params['datasets']
    test_set_x, test_set_y = datasets['test_set']
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.get_value()
    params['datasets'] = None

    results = [('params', params)]
    predictions = []
    for name, model in models:
        print '---------------------------------------', name, '---------------------------------------'

        cpickle_name = task_type + '_' + name + '.save'

        best_validation_loss, test_score = train(
            model=model,
            learning_rate=params['learning_rate'],
            annealing_learning_rate=params['annealing_learning_rate'],
            l1_learning_rate=params['l1_learning_rate'],
            l2_learning_rate=params['l2_learning_rate'],
            n_epochs=params['n_epochs'],
            batch_size=params['batch_size'],
            datasets=datasets,
            name_model=cpickle_name,
            pre_training_epochs=params['pre_training_epochs'],
            pre_train_lr=params['pre_train_lr'],
            k=params['k'],
            noise_rate=params['noise_rate'],
            dropout_rate=params['dropout_rate']
        )

        # Test
        print "Test values test set:"
        predicted_values = predict(
            prediction_set=test_set_x,
            name_model=cpickle_name
        )

        # Euclidean distance
        test_error = np.mean(np.linalg.norm(test_set_y - predicted_values, axis=1))
        print 'Test error (Euclidean distance): ', test_error

        test_error2 = np.mean(np.abs(test_set_y - predicted_values), axis=0)
        print 'Test mean error (X, Y): ', test_error2

        for i in range(len(test_set_y)):
            print test_set_y[i], ' : ', predicted_values[i]

        predicted_values = predict(
            prediction_set=datasets['prediction_set'],
            name_model=cpickle_name
        )
        predictions.append(predicted_values.flatten())

        dict_results = {
            'Name': name,
            'test_score': test_error,
            'test_error_x_y': test_error2,
            'theano_test_score': test_score,
            'best_validation_loss': best_validation_loss,
            'predicted_values': predicted_values,
            'test_set': test_set_y,
            'seed': seed
        }
        results.append(dict_results)
    experiments_path = os.path.join(
        os.path.dirname(sys.argv[0]), 'experiments', experiment_name + '_theano_seed_' + str(seed)
    )
    with open(experiments_path, 'wb') as f:
        cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # Save predictions
    df = pd.DataFrame(
        data=np.array(predictions).transpose(),
        columns=[name for name, model in models]
    )
    prediction_name = os.path.join(os.path.dirname(sys.argv[0]), 'predictions', 'theano_predictions' + '.csv')
    df.to_csv(
        prediction_name,
        sep=',',
        encoding='utf-8'
    )


