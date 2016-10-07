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
    dataset_name = 'cinvestav_testbed.csv'
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
        sklearn_model=KNeighborsRegressor(n_neighbors=3),
        num_output=n_out
    )

    # Create ada boosting
    ada_boosting = SklearnNetwork(
        sklearn_model=GradientBoostingRegressor(n_estimators=100, learning_rate=.1, max_depth=5, loss='ls'),
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
    dataset_name = 'cinvestav_testbed.csv'
    seed = 5
    rgn = numpy.random.RandomState(seed)

    datasets = DatasetManager.read_dataset(
        dataset_name=os.path.join(os.path.dirname(__file__), "dataset", dataset_name),
        shared=True,
        seed=seed,
        expected_output=['result_x', 'result_y'],
        skipped_columns=[],
        label_encoding_columns_name=[],
        train_ratio=.6,
        test_ratio=.2,
        valid_ratio=.2
    )

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
        hidden_layers_sizes=[100],
        n_out=n_out,
        numpy_rng=rgn
    )

    deep_belief_network = DBN(
        n_visible=n_in,
        hidden_layers_sizes=[100],
        n_out=n_out,
        numpy_rng=rgn,
        gaussian_visible=False
    )

    gaussian_deep_belief_network = DBN(
        n_visible=n_in,
        hidden_layers_sizes=[100],
        n_out=n_out,
        numpy_rng=rgn,
        gaussian_visible=True
    )

    models = [
        ('Linear Regression', linear_regression_model),
        ('Deep Belief Network', deep_belief_network),
        ('Gaussian Deep Belief Network', gaussian_deep_belief_network),
        ('Multilayer Perceptron', multilayer_perceptron)
    ]

    params = {
        'learning_rate': .001,
        'annealing_learning_rate': 1,
        'l1_learning_rate': 0.001,
        'l2_learning_rate': 0.001,
        'n_epochs': 100,
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
    theano_experiments()
    sklearn_experiments()
