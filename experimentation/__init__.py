import os
import cPickle
import matplotlib.pyplot as plt
from datasets import DatasetManager


def plot_cost(results, data_name, plot_label):
    plt.figure(plot_label)
    plt.ylabel('Accuracy (m)', fontsize=30)
    plt.xlabel('Epoch', fontsize=30)
    plt.yscale('symlog')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True)
    for i in range(1, 2, 1):

        y, x = zip(*results[i][data_name])
        name = results[i]['Name']
        plt.plot(x, y, label=name, linewidth=5.0)
    plt.legend(fontsize='xx-large')


def get_metrics(test_set_y, predicted_values, model_name):
    for i in xrange(len(predicted_values)):
        print predicted_values[i][1]


if __name__ == '__main__':
    """
    seed = 50
    with open(os.path.join('experimentation',  'cinvestav_testbed_experiment_results_' + str(seed)), 'rb') as f:
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
    seed = 50
    dataset, result = DatasetManager.read_dataset2('test_cleaned_dataset.csv', shared=True, seed=seed)
    with open(os.path.join('trained_models',  'Logistic Regressionbrandeis_university.save'), 'rb') as f:
        model = cPickle.load(f)

    predicted_values = model.predict(dataset)
    get_metrics(
        test_set_y=result,
        predicted_values=predicted_values,
        model_name='Logistic Regression'
    )

