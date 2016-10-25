import os
import theano
import theano.tensor as T
import numpy

from experimentation.experiments import run_theano_experiments, run_experiments_sklearn
from models.regression.deep_models.mlp import MLP
from models.regression.deep_models.dbn import DBN
from datasets import DatasetManager

from models.regression.sklearn_models.sklearn_network import SklearnNetwork
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import feature_selection
from models.regression.sklearn_models.rbf import RBF


def sklearn_experiments():
    dataset_name = 'cinvestav_labeled.csv'
    seed = 5
    datasets = DatasetManager.read_dataset(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", dataset_name),
        shared=False,
        seed=seed,
        expected_output=['result_x', 'result_y'],
        skipped_columns=[],
        label_encoding_columns_name=[],
        sklearn_preprocessing=preprocessing.StandardScaler(with_mean=True, with_std=True),
        sklearn_feature_selection=feature_selection.VarianceThreshold(),
        train_ratio=1,
        test_ratio=0,
        valid_ratio=0
    )

    test_set = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", 'cinvestav_labeled_test.csv'),
        expected_output=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        skipped_columns=[],
        shared=False,
        sklearn_preprocessing=datasets['sklearn_preprocessing'],
        sklearn_feature_selection=datasets['sklearn_feature_selection'],
    )

    datasets['test_set'] = test_set
    datasets['prediction_set'] = datasets['test_set'][0]

    train_set_x, train_set_y = datasets['train_set']
    n_in = train_set_x.shape[1]
    n_out = train_set_y.shape[1]

    # Create Radial Basis Networks
    rbf = RBF(
        input_length=n_in,
        hidden_length=500,
        out_lenght=n_out
    )

    # Create KNN
    knn = SklearnNetwork(
        sklearn_model=KNeighborsRegressor(n_neighbors=10),
        num_output=n_out
    )

    # Create ada boosting
    ada_boosting = SklearnNetwork(
        sklearn_model=GradientBoostingRegressor(n_estimators=1000, learning_rate=.1, max_depth=5, loss='ls'),
        num_output=n_out
    )

    models = [
        ('Ada Boosting', ada_boosting),
        ('Radar', knn),
        ('cRBF', rbf)
    ]

    params = {
        'datasets': datasets
    }
    run_experiments_sklearn(
        models=models,
        seed=seed,
        params=params,
        experiment_name='traditional_algorithms',
        task_type='regression'
    )


def theano_experiments():
    dataset_name = 'cinvestav_labeled.csv'
    seed = 5
    rgn = numpy.random.RandomState(seed)

    datasets = DatasetManager.read_dataset(
        dataset_name=os.path.join(os.path.dirname(__file__), 'dataset', 'meters', dataset_name),
        shared=True,
        seed=seed,
        expected_output=['result_x', 'result_y'],
        skipped_columns=[],
        label_encoding_columns_name=[],
        sklearn_preprocessing=preprocessing.StandardScaler(with_mean=True, with_std=True),
        sklearn_feature_selection=feature_selection.VarianceThreshold(),
        train_ratio=.8,
        test_ratio=0,
        valid_ratio=.2
    )

    test_set = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), 'dataset', 'meters', 'cinvestav_labeled_test.csv'),
        expected_output=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        skipped_columns=[],
        sklearn_preprocessing=datasets['sklearn_preprocessing'],
        sklearn_feature_selection=datasets['sklearn_feature_selection'],
        shared=True
    )

    dataset_unlabeled = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", 'cinvestav_unlabeled.csv'),
        skipped_columns=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        sklearn_preprocessing=datasets['sklearn_preprocessing'],
        sklearn_feature_selection=datasets['sklearn_feature_selection'],
        shared=True
    )

    datasets['test_set'] = test_set
    datasets['dataset_unlabeled'] = dataset_unlabeled
    datasets['prediction_set'] = datasets['test_set'][0].get_value()
    train_set_x, train_set_y = datasets['train_set']

    n_in = train_set_x.get_value().shape[1]
    n_out = train_set_y.get_value().shape[1]

    dnn_tanh_models = get_neural_networks(
        n_in,
        n_out,
        rgn,
        activation_function=T.tanh  # T.nnet.relu
    )

    dnn_relu_models = get_neural_networks(
        n_in,
        n_out,
        rgn,
        activation_function=T.nnet.relu
    )

    dnn_sigmoid_models = get_neural_networks(
        n_in,
        n_out,
        rgn,
        activation_function=T.nnet.sigmoid
    )

    dbn_models = get_dbn(
        n_in,
        n_out,
        rgn,
        gaussian=False
    )

    gdbn_models = get_dbn(
        n_in,
        n_out,
        rgn,
        gaussian=True
    )

    models = []
    models.extend(dnn_relu_models)
    models.extend(dnn_sigmoid_models)
    models.extend(dnn_tanh_models)
    models.extend(gdbn_models)
    models.extend(dbn_models)

    params = {
        'learning_rate': .01,
        'annealing_learning_rate': .99999,
        'l1_learning_rate': 0.01,
        'l2_learning_rate': 0.001,
        'n_epochs': 100,
        'batch_size': 20,
        'pre_training_epochs': 50,
        'pre_train_lr': 0.01,
        'k': 1,
        'datasets': datasets,
        'noise_rate': None,
        'dropout_rate': None
    }

    run_theano_experiments(
        models=models,
        seed=seed,
        params=params,
        experiment_name='all_models_with_out_noise_neither_dropout',
        task_type='regression'
    )


def get_neural_networks(n_in, n_out, rgn, activation_function):
    if activation_function == theano.tensor.nnet.sigmoid:
        activation_function_name = 'sigmoid'
    elif activation_function == T.nnet.relu:
        activation_function_name = 'relu'
    else:
        activation_function_name = 'tanh'

    l = 500
    models = []
    for i in range(1, 11, 1):
        hidden_layers = [l - 50 * x for x in range(0, i)]

        multilayer_perceptron = MLP(
            input=T.matrix('x'),
            n_in=n_in,
            hidden_layers_sizes=hidden_layers,
            n_out=n_out,
            numpy_rng=rgn,
            dropout_rate=None,
            activation_function=activation_function
        )
        models.append(('dnn_layers_' + str(i) + '_func_' + activation_function_name, multilayer_perceptron))
    return models


def get_dbn(n_in, n_out, rgn, gaussian):
    l = 500
    models = []
    for i in range(1, 11, 1):
        hidden_layers = [l - 50 * x for x in range(0, i)]

        gaussian_deep_belief_network = DBN(
            n_visible=n_in,
            hidden_layers_sizes=hidden_layers,
            n_out=n_out,
            numpy_rng=rgn,
            gaussian_visible=gaussian
        )

        models.append(('dbn_layers_' + str(i) + '_gaussian_' + str(gaussian), gaussian_deep_belief_network))
    return models


if __name__ == '__main__':
    # sklearn_experiments()
    theano_experiments()

