import pandas as pd
import os
import numpy as np
import random
import theano

__author__ = 'Gibran Felix'


def read_dataset(dataset_name="dataset_simulation_20.csv", shared=False, seed=None):
    print '... loading data'
    dataset_name = os.path.join(os.path.dirname(__file__), "dataset", dataset_name)

    # Only x and y columns
    result = pd.read_csv(dataset_name, index_col=False, dtype=float, header=0, usecols=["result_x", "result_y"])

    # Skip x, y and row number columns
    dataset = pd.read_csv(dataset_name, dtype=float, header=0, index_col=[0, "result_x", "result_y"])

    """
    # Binarize dataset by ranges
    binary_dataset = []
    for x in dataset.values:
        new_x = []
        for x_i in x:
            x_i_binarize = []
            x_i = abs(int(x_i))
            i = x_i / 100
            x_i = "{0:0100b}".format(0)
            x_i_binarize.extend(x_i)
            if(i < len(x_i_binarize)):
                x_i_binarize[i] = "1"
            #else:
            #    x_i_binarize[len(x_i_binarize) - 1] = "1"

            new_x.extend(x_i_binarize)
        binary_dataset.append(new_x)

    binary_dataset = np.asarray(
        binary_dataset,
        dtype=np.int
    )

    Binarize dataset
    binary_dataset = []
    for x in dataset.values:
        new_x = []
        for x_i in x:
            x_i = abs(int(x_i))
            x_i = "{0:07b}".format(x_i)
            new_x.extend(x_i)
        binary_dataset.append(new_x)

    binary_dataset = np.asarray(
        binary_dataset,
        dtype=np.int
    )
    """

    from sklearn import preprocessing
    dataset = preprocessing.scale(dataset.values)
    # Mean1 = dataset.mean(axis=0)
    # Std1 = dataset.std(axis=0)
    # print Mean1, Std1

    return get_sets(dataset=dataset, result=result.values, train_ratio=.6,
                    test_ratio=.2, valid_ratio=.2, shared=shared, seed=seed)


def get_sets(dataset, result, train_ratio=.6, test_ratio=.2, valid_ratio=.2, shared=False, seed=None):
    if seed is not None:
        random.seed(seed)
    # Shuffle data
    zip_dataset = zip(dataset, result)
    random.shuffle(zip_dataset)

    # Get index
    k_train = int(len(zip_dataset) * train_ratio)
    k_test = int(len(zip_dataset) * test_ratio)
    k_valid = int(len(zip_dataset) * valid_ratio)

    index = range(len(zip_dataset))
    train_index = random.sample(population=index, k=k_train)
    index = list(set(index) - set(train_index))
    test_index = random.sample(population=index, k=k_test)
    index = list(set(index) - set(test_index))
    valid_index = random.sample(population=index, k=k_valid)

    # Get zip sets
    zip_train_set = [zip_dataset[i[1]] for i in enumerate(train_index)]
    zip_test_set = [zip_dataset[i[1]] for i in enumerate(test_index)]
    zip_valid_set = [zip_dataset[i[1]] for i in enumerate(valid_index)]

    # Get sets
    train_set = np.array([x[0] for x in zip_train_set]), np.array([d[1]for d in zip_train_set])
    test_set = np.array([x[0] for x in zip_test_set]), np.array([d[1] for d in zip_test_set])
    valid_set = np.array([x[0] for x in zip_valid_set]), np.array([d[1] for d in zip_valid_set])

    if shared:
        test_set = shared_dataset(test_set)
        valid_set = shared_dataset(valid_set)
        train_set = shared_dataset(train_set)

    return train_set, valid_set, test_set


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(
        np.asarray(
            data_x,
            dtype=theano.config.floatX),
        borrow=borrow)
    shared_y = theano.shared(
        np.asarray(data_y,
                   dtype=theano.config.floatX),
        borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y


if __name__ == "__main__" and __package__ is None:
    __package__ = "datasets.DatasetManager"
    read_dataset(dataset_name="dataset_simulation_20.csv", shared=True)

