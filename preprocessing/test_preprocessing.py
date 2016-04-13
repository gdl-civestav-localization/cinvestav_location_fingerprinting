from datasets.DatasetManager import read_dataset
from preprocessing.manifold import get_LLE, get_TSNE
from scaling import get_gaussian_normalization, get_softmax_normalization, get_logistic_normalization, \
    get_linear_normalization, solve_missing_values
from dimensionality_reduction import get_pca

__author__ = 'Gibran Felix'


def test_manifold_LLE():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set

    dataset = get_gaussian_normalization(dataset)
    dataset = get_LLE(dataset, num_components=2, n_neighbors=80)
    assert dataset is not None


def test_manifold_TSNE():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set

    dataset = get_gaussian_normalization(dataset)
    dataset = get_TSNE(dataset, num_components=2)
    assert dataset is not None


def test_dimensionality_reduction_PCA():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set

    dataset, explained_variance_ratio_ = get_pca(dataset, num_components=2)

    assert dataset is not None
    assert explained_variance_ratio_ is not None


def test_scaling_softmax():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set
    dataset = get_softmax_normalization(dataset)
    assert dataset is not None


def test_scaling_logistic():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set
    dataset = get_logistic_normalization(dataset)
    assert dataset is not None


def test_scaling_gaussian():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set
    dataset = get_gaussian_normalization(dataset)
    assert dataset is not None


def test_scaling_linear():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set
    dataset = get_linear_normalization(dataset)
    assert dataset is not None


def test_solve_missing_values():
    train_set, valid_set, test_set = read_dataset(dataset_name='dataset_simulation_zero_iterations.csv')
    dataset, result = train_set
    dataset = solve_missing_values(dataset)
    assert dataset is not None


if __name__ == "__main__":
    test_manifold_LLE()
    test_manifold_TSNE()
    test_dimensionality_reduction_PCA()
    test_scaling_softmax()
    test_scaling_logistic()
    test_scaling_gaussian()
    test_scaling_linear()
    test_solve_missing_values()
