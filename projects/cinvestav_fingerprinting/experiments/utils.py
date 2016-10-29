import numpy as np
import pandas as pd


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
show_results = True


def get_metrics(results, latex=False):
    lst_error = [0]
    metrics = []
    for i in results:
        model_name = i['Name']
        predicted_values = i['predicted_values']
        test_set = i['test_set']

        test_error = np.linalg.norm(test_set - predicted_values, axis=1)
        test_error = test_error.astype(dtype=np.float64)

        min_value = test_error.min()
        max_value = test_error.max()
        avg_value = test_error.mean()
        std_value = test_error.std()
        test_error.sort()

        metrics.append([
            model_name,
            min_value,
            max_value,
            avg_value,
            std_value,
            test_error[int(len(test_error) * .90)],
            test_error[int(len(test_error) * .70)],
            test_error[int(len(test_error) * .50)]
        ])
        lst_error.append(avg_value)

    columns = [
        '',
        'Min',
        'Max',
        'Mean',
        'Std',
        '90',
        '70',
        '50'
    ]
    dataframe = pd.DataFrame(
        data=metrics,
        columns=columns
    )
    dataframe = dataframe.T
    dataframe.columns = dataframe.iloc[0]
    dataframe = dataframe[1:]
    if show_results:
        if latex:
            print dataframe.to_latex()
        else:
            print dataframe.head(10)
    return lst_error