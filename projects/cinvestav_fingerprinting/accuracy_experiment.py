import os
import theano.tensor as T
import numpy

from experimentation.experiments import run_theano_experiments, run_experiments_sklearn
from models.regression.deep_models.linear_regression import LinearRegression
from models.regression.deep_models.mlp import MLP
from models.regression.deep_models.dbn import DBN
from datasets import DatasetManager

from models.regression.sklearn_models.sklearn_network import SklearnNetwork
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
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
        train_ratio=.8,
        test_ratio=.2,
        valid_ratio=0
    )

    test_set = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", 'cinvestav_labeled_test.csv'),
        expected_output=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        skipped_columns=[],
        mean=datasets['mean'],
        std=datasets['std'],
        shared=False,
        feature_selection=datasets['feature_selection']
    )

    # datasets['test_set'] = test_set
    datasets['prediction_set'] = datasets['test_set'][0]

    train_set_x, train_set_y = datasets['train_set']
    n_in = train_set_x.shape[1]
    n_out = train_set_y.shape[1]

    # Create Radial Basis Networks
    rbf = RBF(
        input_length=n_in,
        hidden_length=50,
        out_lenght=n_out
    )

    # Create KNN
    knn = SklearnNetwork(
        sklearn_model=KNeighborsRegressor(n_neighbors=5),
        num_output=n_out
    )

    # Create ada boosting
    ada_boosting = SklearnNetwork(
        sklearn_model=GradientBoostingRegressor(n_estimators=1000, learning_rate=.1, max_depth=5, loss='ls'),
        num_output=n_out
    )

    models = [
        ('Ada Boosting', ada_boosting),
        ('K Neighbors Regressor', knn)
        # ('cRBF', rbf)
    ]

    params = {
        'datasets': datasets
    }
    run_experiments_sklearn(
        models=models,
        seed=seed,
        params=params,
        task_type='regression'
    )


def theano_experiments():
    dataset_name = 'cinvestav_labeled.csv'
    seed = 5
    rgn = numpy.random.RandomState(seed)

    datasets = DatasetManager.read_dataset(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", dataset_name),
        shared=True,
        seed=seed,
        expected_output=['result_x', 'result_y'],
        skipped_columns=[],
        label_encoding_columns_name=[],
        train_ratio=.8,
        test_ratio=0,
        valid_ratio=.2
    )

    test_set = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", 'cinvestav_labeled_test.csv'),
        expected_output=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        skipped_columns=[],
        mean=datasets['mean'],
        std=datasets['std'],
        shared=True,
        feature_selection=datasets['feature_selection']
    )

    dataset_unlabeled = DatasetManager.get_prediction_set(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", 'cinvestav_unlabeled.csv'),
        skipped_columns=['result_x', 'result_y'],
        label_encoding_columns_name=[],
        mean=datasets['mean'],
        std=datasets['std'],
        shared=True,
        feature_selection=datasets['feature_selection']
    )

    datasets['test_set'] = test_set
    datasets['dataset_unlabeled'] = dataset_unlabeled
    datasets['prediction_set'] = datasets['test_set'][0].get_value()
    train_set_x, train_set_y = datasets['train_set']

    n_in = train_set_x.get_value().shape[1]
    n_out = train_set_y.get_value().shape[1]

    x = T.matrix('x')

    linear_regression_model = LinearRegression(
        input=x,
        n_in=n_in,
        n_out=n_out
    )

    multilayer_perceptron = MLP(
        input=x,
        n_in=n_in,
        hidden_layers_sizes=[1000, 700, 500, 400, 300],
        n_out=n_out,
        numpy_rng=rgn,
        activation_function=T.nnet.relu  # T.tanh
    )

    deep_belief_network = DBN(
        n_visible=n_in,
        hidden_layers_sizes=[100, 70, 50, 40, 30],
        n_out=n_out,
        numpy_rng=rgn,
        gaussian_visible=False
    )

    gaussian_deep_belief_network = DBN(
        n_visible=n_in,
        hidden_layers_sizes=[100, 70, 50, 40, 30],
        n_out=n_out,
        numpy_rng=rgn,
        gaussian_visible=True
    )

    models = [
        ('Gaussian Deep Belief Network', gaussian_deep_belief_network),
        ('Deep Belief Network', deep_belief_network),
        ('Multilayer Perceptron', multilayer_perceptron),
        ('Linear Regression', linear_regression_model)
    ]

    params = {
        'learning_rate': .01,
        'annealing_learning_rate': .9999,
        'l1_learning_rate': 0.01,
        'l2_learning_rate': 0.001,
        'n_epochs': 500,
        'batch_size': 20,
        'pre_training_epochs': 50,
        'pre_train_lr': 0.001,
        'k': 1,
        'datasets': datasets
    }

    run_theano_experiments(
        models=models,
        seed=seed,
        params=params,
        task_type='regression'
    )


if __name__ == '__main__':
    # sklearn_experiments()
    theano_experiments()

