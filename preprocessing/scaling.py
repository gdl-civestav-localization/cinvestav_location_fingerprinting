import numpy as np
from sklearn.preprocessing import Imputer

__author__ = 'Gibran Felix'


# Perform a softmax scaling
def get_softmax_normalization(data=None, t=1.0):
    if not data:
        data = []

    e = np.exp(np.array(data) / t)
    dist = e / np.sum(e)
    return dist


# Perform a logistic scaling
def get_logistic_normalization(data=None):
    if not data:
        data = []

    data = 1 / (1 + np.exp(-1 * np.array(data)))
    return data


# Perform a gaussian scaling
def get_gaussian_normalization(data=np.array([])):
    data = (data - np.mean(data, 0)) / np.std(data, 0)
    return data


# Perform a linear scaling
def get_linear_normalization(data=np.array([])):
    min_value = np.min(data, 0)
    max_value = np.max(data, 0)
    data = (data - min_value) / (max_value - min_value)
    return data


# Solve missing values
def solve_missing_values(data=None):
    if not data:
        data = []

    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit_transform(data)
    return data



