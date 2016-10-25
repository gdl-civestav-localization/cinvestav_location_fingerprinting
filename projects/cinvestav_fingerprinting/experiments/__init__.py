import os
import cPickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
show_results = True


def plot_cost(results, data_name, plot_label):
    plt.figure(plot_label)
    plt.ylabel('Accuracy (m)', fontsize=30)
    plt.xlabel('Epoch', fontsize=30)
    plt.yscale('symlog')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    for i in range(2, 5, 1):
        y, x = zip(*results[i][data_name])
        name = results[i]['Name']
        plt.plot(x, y, label=name, linewidth=5.0)
    plt.legend(fontsize='xx-large')


def get_metrics(results):
    lst_error = []
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
            test_error[int(len(test_error) * .95)],
            test_error[int(len(test_error) * .66)],
            test_error[int(len(test_error) * .50)]
        ])
        lst_error.append(avg_value)
        # print_metrics(euclidean_error)

    columns = [
        '',
        'Min',
        'Max',
        'Mean',
        'Std',
        '95',
        '66',
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
        print dataframe.head(10)

    return lst_error


if __name__ == '__main__':
    """
    seed = 40
    with open(os.path.join('experiments',  'experiment_results_' + str(seed)), 'rb') as f:
        results = cPickle.load(f)

    plot_cost(
        results=results,
        data_name='cost_train',
        plot_label='Cost on train phase')
    plot_cost(
        results=results,
        data_name='cost_valid',
        plot_label='Cost on valid phase')
    plot_cost(
        results=results,
        data_name='cost_test',
        plot_label='Cost on test phase')
    plt.show()
    """

    with open(os.path.join(os.path.dirname(__file__), 'all_models_with_out_noise_neither_dropout_theano_seed_5'), 'rb') as f:
        results = cPickle.load(f)
    results.pop(0)  # Remove params

    lst_error = get_metrics(
        results=results
    )

    plt.figure("Depth experiement")
    plt.ylabel('Accuracy (m)', fontsize=30)
    plt.xlabel('Layers', fontsize=30)
    # plt.yscale('symlog')
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(lst_error, linewidth=5.0)
    plt.show()
    print lst_error
