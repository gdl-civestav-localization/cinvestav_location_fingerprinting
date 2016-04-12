import cPickle
import os
import theano.tensor as T
import numpy

from deep_models.dbn import DBN
from deep_models.deep_learning import train, predict
from deep_models.linear_regression import LinearRegression
from datasets import DatasetManager
from deep_models.mlp import MLP


def run_experiments(models, seed, params):
    datasets = params['datasets']
    params['datasets'] = None
    results = [('params', params)]
    for model, name in models:
        cpickle_name = name + '_regressor_RSSI20.save'

        lst_cost_test, lst_cost_valid, lst_cost_train = train(
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
            k=params['k']
        )

        # load the saved model
        # with open(os.path.join('trained_models', cpickle_name), 'rb') as f:
        # model = cPickle.load(f)

        predicted_values = predict(
            datasets=datasets,
            name_model=cpickle_name
        )

        dict_results = {
            'Name': name,
            'cost_train': lst_cost_train,
            'cost_valid': lst_cost_valid,
            'cost_test': lst_cost_test,
            'predicted_values': predicted_values,
            'seed': seed
        }
        results.append(dict_results)

        # get_metrics(
        #     test_set_y=datasets[2][1].get_value(),
        #     predicted_values=predicted_values,
        #     model_name=name
        # )
    with open(os.path.join('experiments',  'correction2_experiment_results_' + str(seed)), 'wb') as f:
        cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)


def run_depth_experiment():
    seed = 9
    datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=True, seed=seed)
    train_set_x, train_set_y = datasets[0]

    n_in = train_set_x.get_value().shape[1]
    n_out = train_set_y.get_value().shape[1]

    x = T.matrix('x')
    rng = numpy.random.RandomState(60)

    mlp_model1 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model2 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model3 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model4 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model5 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model6 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model7 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model8 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model9 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    mlp_model10 = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
        n_out=n_out,
        activation_function=T.tanh
    )

    linear_regressor_model = LinearRegression(
        input=x,
        n_in=n_in,
        n_out=n_out
    )

    models = [
        (linear_regressor_model, 'SLN'),
        (mlp_model1, 'DNN-DEPTH 1'),
        (mlp_model2, 'DNN-DEPTH 2'),
        (mlp_model3, 'DNN-DEPTH 3'),
        (mlp_model4, 'DNN-DEPTH 4'),
        (mlp_model5, 'DNN-DEPTH 5'),
        (mlp_model6, 'DNN-DEPTH 6'),
        (mlp_model7, 'DNN-DEPTH 7'),
        (mlp_model8, 'DNN-DEPTH 8'),
        (mlp_model9, 'DNN-DEPTH 9'),
        (mlp_model10, 'DNN-DEPTH 10')
    ]

    params = {
        'learning_rate': 0.001,
        'annealing_learning_rate': 1,
        'l1_learning_rate': 0.001,
        'l2_learning_rate': 0.0001,
        'n_epochs': 10000,
        'batch_size': 600,
        'pre_training_epochs': 50,
        'pre_train_lr': 0.001,
        'k': 1,
        'datasets': datasets
    }
    run_experiments(models=models, seed=seed, params=params)

if __name__ == '__main__':
    run_depth_experiment()
    """
    # for seed in range(20, 30, 1):
    for seed in range(50, 51, 1):
        datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=True, seed=seed)
        train_set_x, train_set_y = datasets[0]

        n_in = train_set_x.get_value().shape[1]
        n_out = train_set_y.get_value().shape[1]

        x = T.matrix('x')
        rng = numpy.random.RandomState(1234)

        linear_regressor_model = LinearRegression(
            input=x,
            n_in=n_in,
            n_out=n_out
        )

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

        models = [
            (linear_regressor_model, 'Linear Regression'),
            (mlp_model, 'DNN'),
            (dbn_model, 'DBN'),
            (gbrbm_dbn_model, 'GB-DBN')
        ]

        params = {
            'learning_rate': 0.01,
            'annealing_learning_rate': 1,
            'l1_learning_rate': 0.001,
            'l2_learning_rate': 0.0001,
            'n_epochs': 10000,
            'batch_size': 600,
            'pre_training_epochs': 50,
            'pre_train_lr': 0.001,
            'k': 1,
            'datasets': datasets
        }
        run_experiments(models=models, seed=seed, params=params)
        """
