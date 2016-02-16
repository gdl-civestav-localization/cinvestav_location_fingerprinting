import pandas as pd
import os
__author__ = 'Gibran Felix'


def read_dataset(dataset_name="dataset_simulation_20.csv"):
    dataset_name = os.path.join(os.path.dirname(__file__), "dataset", dataset_name)

    # Only x and y columns
    result = pd.read_csv(dataset_name, index_col=False, dtype=float, header=0, usecols=["result_x", "result_y"])

    # Skip x, y and row number columns
    dataset = pd.read_csv(dataset_name, dtype=float, header=0, index_col=[0, "result_x", "result_y"])

    return dataset.values, result.values
