import RadialBasisFuncionNetwork
import numpy as np
import random
import DatasetManager
import Preprocessing
import SVM
import theano
import DeepLearning.rbm as rbm
import cPickle

from matplotlib import pyplot as plt

__author__ = 'Gibran'

random.seed(23)


def get_sets(ratio=.8):
    result, dataset = DatasetManager.read_dataset()

    # Normalize
    dataset = Preprocessing.get_gaussian_normalization(dataset)
    # dataset, s = Preprocessing.get_pca(dataset, num_components=6)
    # dataset = Preprocessing.get_TSNE(dataset, num_components=6)
    result = (result - np.mean(result, 0))

    # Shuffle data
    zip_dataset = zip(result, dataset)
    random.shuffle(zip_dataset)

    # Get index
    index = range(len(zip_dataset))
    train_index = random.sample(index, int(len(index)*ratio))
    test_index = list(set(index) - set(train_index))

    # Get zip sets
    zip_train_set = [zip_dataset[i[1]] for i in enumerate(train_index)]
    zip_test_set = [zip_dataset[i[1]] for i in enumerate(test_index)]

    # Get sets
    train_set = (np.array([d[0] for d in zip_train_set]), np.array([x[1] for x in zip_train_set]))
    test_set = ([d[0] for d in zip_test_set], [x[1] for x in zip_test_set])

    return train_set, test_set


def run_neuronal_perception(eta=.1, theta=.1, iteration=1000, a=10, b=.01, momentum_costant=.1):
    train_set, test_set = get_sets(.8)
    input_layer = len(train_set[1][0])

    # Create the NN
    n = SVM.NeuronalNetworks(input_layer, num_output=2)

    # Train it with some patterns
    x = train_set[1]
    y = train_set[0]
    errors = n.train(x, y)

    # Test Neuronal network
    x = test_set[1]
    y =  test_set[0]
    results, error  = n.test(x, y)

    print error

    # Plot desired output
    plt.figure("Plot desired output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*y)[0], zip(*y)[1], 'bx', linewidth=4)

    # Plot output
    plt.figure("Plot output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*results)[0], zip(*results)[1], 'rx', linewidth=4)
    plt.show()


def run_dbn():
    train_set, test_set = get_sets(.8)

    train_set_x = theano.shared(np.asarray(train_set[1], dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(np.asarray(train_set[0], dtype=theano.config.floatX), borrow=True)

    test_set_x = theano.shared(np.asarray(test_set[1], dtype=theano.config.floatX), borrow=True)
    test_set_y = theano.shared(np.asarray(test_set[0], dtype=theano.config.floatX), borrow=True)

    print "Training"
    rbm.train_rbm(
        dataset=(train_set_x, train_set_y),
        learning_rate=0.1,
        training_epochs=50,
        batch_size=20,
        output_folder='rbm_plots_location',
        n_hidden=500,
        n_visible=len(test_set_x.get_value(borrow=True)[0]),
        name_model='rbm_location.save')

    print "Testing"
    train_set_x = rbm.test_rbm(
        dataset=(train_set_x, train_set_y),
        plot_every=10,
        n_samples=1,
        output_folder='rbm_plots_location',
        name_model='rbm_location.save'
    )

    test_set_x = rbm.test_rbm(
        dataset=(test_set_x, test_set_y),
        plot_every=10,
        n_samples=1,
        output_folder='rbm_plots_location',
        name_model='rbm_location.save'
    )

    dataset = ((train_set_y.get_value(borrow=True), train_set_x),(test_set_y.get_value(borrow=True), test_set_x))
    with open("x.save", 'wb') as f:
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)


def run_svm():
    train_set, test_set = get_sets(.8)

    # Create Support Vector Machine
    svm = SVM.SVM(2)

    # Train SVM
    x = train_set[1]
    y = train_set[0]
    svm.train(x, y)

    # Test SVM
    x = test_set[1]
    y =  test_set[0]
    results, error = svm.test(x, y)

    print "Error: " + str(error)

    # Plot desired output
    plt.figure("Plot desired output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*y)[0], zip(*y)[1], 'bx', linewidth=4)

    # Plot output
    plt.figure("Plot output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*results)[0], zip(*results)[1], 'rx', linewidth=4)
    plt.show()


def run_adaboosting():
    # with open("x.save", 'rb') as f:
    #     train_set, test_set = cPickle.load(f)

    train_set, test_set = get_sets(.8)

    # Create Support Vector Machine
    ada_boosting = SVM.AdaBoosting(2)

    # Train SVM
    x = train_set[1]
    y = train_set[0]
    ada_boosting.train(x, y)

    # Test Ada Boosting
    x = test_set[1]
    y = test_set[0]
    result, error = ada_boosting.test(x, y)

    print "Error: " + str(error)

    # Plot desired output
    plt.figure("Plot desired output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*y)[0], zip(*y)[1], 'bx', linewidth=4)

    # Plot output
    plt.figure("Plot output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*result)[0], zip(*result)[1], 'rx', linewidth=4)
    plt.show()


def run_radial_basis_function_network(hidden, lamda=0):
    train_set, test_set = get_sets()
    input_layer = len(train_set[1][0])

    # Create Radial Basis Networks
    rbf = RadialBasisFuncionNetwork.RBF(input_layer, hidden, 2)

    # Train
    x = train_set[1]
    d = train_set[0]
    errors = rbf.train(x, d, lamda)

    # Test
    x = test_set[1]
    d = test_set[0]
    result = rbf.test(x)
    error = rbf.cost_function(d, result)

    # Print error
    print "Error: " + str(error)

    # Plot desired output
    plt.figure("Plot desired output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*d)[0], zip(*d)[1], 'bx', linewidth=4)

    # Plot output
    plt.figure("Plot output")
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.plot(zip(*result)[0], zip(*result)[1], 'rx', linewidth=4)
    plt.show()


if __name__ == '__main__':
    # run_svm()
    run_adaboosting()
    # run_neuronal_perception(eta=.01, theta=.1, iteration=1000, a=50, b=.001, momentum_costant=.08)
    # run_dbn()
    # run_radial_basis_function_network(hidden=200, lamda=0)
    # print "Listo"
    # [e for e in n
    # .nn.edges_iter(nbunch=[0,1],data=True)]