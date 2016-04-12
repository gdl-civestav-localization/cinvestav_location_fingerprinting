import cPickle
import os
from datasets import DatasetManager
from models.svm import SVM
from models.rbf import RBF
from models.knn import KNN
from models.ada_boosting import AdaBoosting

__author__ = 'Gibran'


def run_experiments(models, seed, params):
    datasets = params['datasets']
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    results = [('params', params)]
    for model, name in models:
        print name, 'Seed', seed
        # Train
        model.train(train_set_x, train_set_y)

        # Test
        predicted_values = model.test(test_set_x)

        dict_results = {
            'Name': name,
            'predicted_values': predicted_values,
            'seed': seed
        }
        results.append(dict_results)
    with open(os.path.join('deep_models', 'experiments',  'no_deep_experiment_results_' + str(seed)), 'wb') as f:
        cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for seed in range(20, 30, 1):
        datasets = DatasetManager.read_dataset('dataset_simulation_20.csv', shared=False, seed=seed)
        train_set_x, train_set_y = datasets[0]

        n_in = train_set_x.shape[1]
        n_out = train_set_y.shape[1]

        # Create Radial Basis Networks
        rbf = RBF(
            input_length=n_in,
            hidden_length=200,
            out_lenght=n_out
        )

        # Create support vector machine
        svm = SVM(n_out)

        # Create KNN
        knn = KNN(n_out)

        # Create ada boosting
        ada_boosting = AdaBoosting(n_out)

        models = [
            (knn, 'RADAR'),
            (svm, 'SVM'),
            (rbf, 'cRBF'),
            (ada_boosting, 'Ada Boosting')
        ]

        params = {
            'datasets': datasets
        }
        run_experiments(models=models, seed=seed, params=params)
