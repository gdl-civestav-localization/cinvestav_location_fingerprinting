import os
import cPickle
import matplotlib.pyplot as plt
import numpy as np

from datasets import DatasetManager


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


def get_metrics(test_set_y, predicted_values, model_name):
    euclidean_error = np.array([np.linalg.norm(y - d) for d, y in zip(test_set_y, predicted_values)])
    # abs_error = np.array([np.sum(abs(y - d)) for d, y in zip(test_set_y, predicted_values)])
    # ecm_error = np.array([np.sum((y - d) ** 2) / len(d) for d, y in zip(test_set_y, predicted_values)])

    def print_metrics(values):
        min_value = values.min()
        max_value = values.max()
        avg_value = values.mean()
        std_value = values.std()
        values.sort()
        print '----------------------------------------'
        print model_name
        print '----------------------------------------'
        print 'Min:', min_value
        print 'Max:', max_value
        print 'Mean:', avg_value
        print 'Std:', std_value
        print '99%:', values[int(len(values) * .99)]
        print '66%:', values[int(len(values) * .66)]
        print '50%:', values[int(len(values) * .50)]

    print_metrics(euclidean_error)
    return euclidean_error
    # print_metrics(abs_error)
    # print_metrics(ecm_error)


if __name__ == '__main__':

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
    seed = 9
    datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=True, seed=seed)
    with open(os.path.join('experiments',  'correction2_experiment_results_' + str(seed)), 'rb') as f:
        results = cPickle.load(f)

    lst_error = []
    for i in range(1, 12, 1):
        lst_predicted = []
        lst_test = []

        lst_test.extend(datasets[2][1].get_value())
        lst_predicted.extend(results[i]['predicted_values'])
        euclidean_error = get_metrics(
            test_set_y=lst_test,
            predicted_values=lst_predicted,
            model_name=results[i]['Name']
        )
        lst_error.append(euclidean_error.mean())

    plt.figure("Depth experiement")
    plt.ylabel('Accuracy (m)', fontsize=30)
    plt.xlabel('Layers', fontsize=30)
    # plt.yscale('symlog')
    plt.grid(True)

    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(lst_error, linewidth=5.0)
    plt.show()
    print lst_error
    """