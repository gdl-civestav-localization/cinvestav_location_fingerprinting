import cPickle
import matplotlib.pyplot as plt
import glob
from projects.cinvestav_fingerprinting.experiments.utils import get_metrics


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


def plot_per_experiment():
    for filename in glob.glob('*all_model*'):
        with open(filename, 'rb') as f:
            results = cPickle.load(f)
        results.pop(0)  # Remove params

        dnn_relu_models = results[0:10]
        dnn_sigmoid_models = results[10:20]
        dnn_tanh_models = results[20:30]
        gdbn_models = results[30:40]
        dbn_models = results[40:50]

        plt.figure(filename)
        plt.ylabel('Accuracy (m)', fontsize=30)
        plt.xlabel('Layers', fontsize=30)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.plot(get_metrics(results=dnn_relu_models), label='DNN Relu', linewidth=5.0)
        plt.plot(get_metrics(results=dnn_sigmoid_models), label='DNN Sigmoid', linewidth=5.0)
        plt.plot(get_metrics(results=dnn_tanh_models), label='DNN Tanh', linewidth=5.0)
        plt.plot(get_metrics(results=gdbn_models), label='GR-DBN', linewidth=5.0)
        plt.plot(get_metrics(results=dbn_models), label='DBN', linewidth=5.0)
        plt.legend(fontsize='large')
    plt.show()


def plot_all_experiments():
    plt.figure("Depth experiement")
    plt.ylabel('Accuracy (m)', fontsize=30)
    plt.xlabel('Layers', fontsize=30)
    plt.ylim([0, 10])
    plt.yticks(range(0, 11))
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    for filename in glob.glob('*seed*'):
        with open(filename, 'rb') as f:
            results = cPickle.load(f)
        results.pop(0)  # Remove params

        lst_error = get_metrics(
            results=results
        )
        plt.plot(lst_error, label=filename, linewidth=5.0)
    plt.legend(fontsize='large')
    plt.show()


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
    # plot_all_experiments()

    plot_per_experiment()
