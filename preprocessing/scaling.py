import numpy as np

__author__ = 'Gibran Felix'


def get_softmax_normalization(data, t=1.0):
    """
    Perform a softmax scaling
    Parameters
    ----------
    data: Values to transform
    t: Value
    """
    e = np.exp(np.array(data) / t)
    dist = e / np.sum(e)
    return dist


def get_logistic_normalization(data):
    """
    Perform a logistic scaling
    Parameters
    ----------
    data: Values to transform
    """
    data = 1 / (1 + np.exp(-1 * np.array(data)))
    return data


def get_gaussian_normalization(data, mean=None, std=None):
    """
    Perform a gaussian scaling
    Parameters
    ----------
    data: Values to transform
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    data = (data - mean) / std
    return data, mean, std


def get_linear_normalization(data):
    """
    Perform a linear scaling
    Parameters
    ----------
    data: Values to transform
    """
    min_value = np.min(data, 0)
    max_value = np.max(data, 0)
    data = (data - min_value) / (max_value - min_value)
    return data


def solve_missing_values(data):
    """
    Solve missing values
    Parameters
    ----------
    data: Values to remove missing values
    """
    from sklearn.preprocessing import Imputer

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit_transform(data)
    return data



