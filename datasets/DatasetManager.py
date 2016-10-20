import pandas as pd
import os
import numpy as np
import theano
import random
from preprocessing.scaling import get_gaussian_normalization

__author__ = 'Gibran Felix'


def get_prediction_set(
        dataset_name,
        skipped_columns=[],
        expected_output=[],
        one_hot_encoding_columns_name=[],
        label_encoding_columns_name=[],
        sklearn_preprocessing=None,
        sklearn_feature_selection=None,
        shared=False
):
    dataset_name = os.path.join(os.path.dirname(__file__), "dataset", dataset_name)

    dataframe = pd.read_csv(
        dataset_name,
        index_col=False,
        dtype=object,
        header=0
    )

    # Skipped columns
    skipped_columns.append(0)  # Panda index row
    skipped_columns.extend(expected_output)
    skipped_columns.extend(one_hot_encoding_columns_name)
    skipped_columns.extend(label_encoding_columns_name)

    # Get dataset
    dataset = pd.read_csv(
        dataset_name,
        header=0,
        dtype=np.float32,
        index_col=skipped_columns
    ).fillna(-110).replace([0], [-110]).values

    # One hot encoding
    if len(one_hot_encoding_columns_name) > 0:
        one_hot_encoding_data, one_hot_encoding_columns_name = one_hot_encoding(
            panda_dataframe=dataframe,
            column_names=one_hot_encoding_columns_name
        )
        dataset = np.concatenate(
            (
                dataset,
                one_hot_encoding_data
            ),
            axis=1
        )

    # Label encoder
    if len(label_encoding_columns_name) > 0:
        label_encoding_data = label_encoder(
            panda_dataframe=dataframe,
            column_names=label_encoding_columns_name
        )
        dataset = np.concatenate(
            (
                dataset,
                label_encoding_data
            ),
            axis=1
        )

    # To force the dataset to be numeric, throw error if is not completely numeric
    dataset = dataset.astype(np.float32, copy=False)

    if sklearn_feature_selection is not None:
        dataset = sklearn_feature_selection.transform(dataset)

    # Normalized dataset
    if sklearn_preprocessing is not None:
        dataset = sklearn_preprocessing.transform(dataset)

    if shared:
        dataset = theano.shared(
            np.asarray(
                dataset,
                dtype=theano.config.floatX
            ),
            borrow=True
        )

    if len(expected_output) == 0:
        return dataset
    else:
        # Get expected values
        result = dataframe.as_matrix(columns=expected_output)
        result = result.astype(np.float32, copy=False)
        if shared:
            result = theano.shared(
                np.asarray(
                    result,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        return dataset, result


def read_dataset(
        dataset_name="dataset.csv",
        shared=False,
        seed=None,
        expected_output=[],
        skipped_columns=[],
        one_hot_encoding_columns_name=[],
        label_encoding_columns_name=[],
        sklearn_preprocessing=None,
        sklearn_feature_selection=None,
        train_ratio=.6,
        test_ratio=.2,
        valid_ratio=.2,
        task_type='regression'
):
    print '... loading data', dataset_name
    dataframe = pd.read_csv(
        dataset_name,
        index_col=False,
        dtype=object,
        header=0
    )

    # Skipped columns
    skipped_columns.append(0)  # Panda index row
    skipped_columns.extend(expected_output)
    skipped_columns.extend(one_hot_encoding_columns_name)
    skipped_columns.extend(label_encoding_columns_name)

    # Get dataset
    dataset = pd.read_csv(
        dataset_name,
        header=0,
        index_col=skipped_columns,
        dtype=np.float32
    ).fillna(-110).replace([0], [-110]).values

    # Get expected values
    result = dataframe.as_matrix(columns=expected_output)

    # One hot encoding
    if len(one_hot_encoding_columns_name) > 0:
        one_hot_encoding_data, one_hot_encoding_columns_name = one_hot_encoding(
            panda_dataframe=dataframe,
            column_names=one_hot_encoding_columns_name
        )
        dataset = np.concatenate(
            (
                dataset,
                one_hot_encoding_data
            ),
            axis=1
        )

    # Label encoder
    if len(label_encoding_columns_name) > 0:
        label_encoding_data = label_encoder(
            panda_dataframe=dataframe,
            column_names=label_encoding_columns_name
        )
        dataset = np.concatenate(
            (
                dataset,
                label_encoding_data
            ),
            axis=1
        )

    # To force the dataset to be numeric, throw error if is not completely numeric
    dataset = dataset.astype(np.float32, copy=False)
    result = result.astype(np.float32, copy=False)

    if sklearn_feature_selection is not None:
        dataset = sklearn_feature_selection.fit_transform(dataset, result[:,0:1])

    # Normalized dataset
    if sklearn_preprocessing is not None:
        dataset = sklearn_preprocessing.fit_transform(dataset)

    train_set, valid_set, test_set = get_sets(
        dataset=dataset,
        result=result,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        shared=shared,
        seed=seed,
        task_type=task_type
    )

    return {
        'train_set': train_set,
        'valid_set': valid_set,
        'test_set': test_set,
        'sklearn_preprocessing': sklearn_preprocessing,
        'sklearn_feature_selection': sklearn_feature_selection
    }


def get_sets(
        dataset,
        result,
        train_ratio=.6,
        test_ratio=.2,
        valid_ratio=.2,
        shared=False,
        seed=None,
        task_type='regression'
):
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
    train_set = np.array([x[0] for x in zip_train_set]), np.array([d[1] for d in zip_train_set])
    test_set = np.array([x[0] for x in zip_test_set]), np.array([d[1] for d in zip_test_set])
    valid_set = np.array([x[0] for x in zip_valid_set]), np.array([d[1] for d in zip_valid_set])

    if shared:
        test_set = shared_dataset(test_set, task_type=task_type)
        valid_set = shared_dataset(valid_set, task_type=task_type)
        train_set = shared_dataset(train_set, task_type=task_type)

    return train_set, valid_set, test_set


def shared_dataset(data_xy, borrow=True, task_type='regression'):
    import theano
    data_x, data_y = data_xy
    shared_x = theano.shared(
        np.asarray(
            data_x,
            dtype=theano.config.floatX
        ),
        borrow=borrow)
    shared_y = theano.shared(
        np.asarray(
            data_y,
            dtype=theano.config.floatX
        ),
        borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    import theano.tensor as T

    if task_type == 'regression':
        return shared_x, shared_y
    elif task_type == 'classification':
        return shared_x, T.cast(shared_y, 'int32')


def one_hot_encoding(panda_dataframe, column_names):
    data = []
    columns = []

    for c in column_names:
        column_data = panda_dataframe[c].fillna('missing').values
        column_dataframe = pd.get_dummies(column_data)

        new_column_data = column_dataframe.values
        columns.extend(column_dataframe.columns)
        data.append(new_column_data)
    # Transform to matrix form
    data = np.concatenate(
        (
            data
        ),
        axis=1
    ).astype(np.float64)
    return data, columns


def label_encoder(panda_dataframe, column_names):
    from sklearn import preprocessing

    data = []
    for c in column_names:
        column_data = panda_dataframe[c].fillna('missing').values

        encoder = preprocessing.LabelEncoder()
        encoder.fit(column_data)
        new_column_data = encoder.transform(column_data).reshape(-1, 1)
        data.append(new_column_data)
    # Transform to matrix form
    data = np.concatenate(
        (
            data
        ),
        axis=1
    ).astype(np.float64)
    return data


if __name__ == "__main__":
    read_dataset(dataset_name="cleaned_dataset.csv", shared=False)
    read_dataset(dataset_name="cleaned_dataset.csv", shared=True)
