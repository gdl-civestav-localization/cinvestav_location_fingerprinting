from rbm import RBM
import theano.tensor as T

import timeit
import os
from theano.tensor.shared_randomstreams import RandomStreams
import numpy
try:
    import PIL.Image as Image
except ImportError:
    import Image
import theano
from utils import load_data, tile_raster_images
import cPickle


class GBRBM(RBM):
    """Gaussian Bernoulli Restricted Boltzmann Machine (RBM)  """
    def __init__(self, input=None, n_visible=784, n_hidden=500,
                 W=None, h_bias=None, v_bias=None, numpy_rng=None, theano_rng=None,
                 name='grbm', W_r=None, dropout=0, dropconnect=0, transpose=False, activation=T.nnet.sigmoid):
        """
        GBRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa).
        It initialize parent class (RBM).

        :param input: None for standalone RBMs or symbolic variable if RBM is
            part of a larger graph.
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param W: None for standalone RBMs or symbolic variable pointing to a
            shared weight matrix in case RBM is part of a DBN network; in a DBN,
            the weights are shared between RBMs and layers of a MLP
        :param h_bias: None for standalone RBMs or symbolic variable pointing
            to a shared hidden units bias vector in case RBM is part of a
            different network
        :param v_bias: None for standalone RBMs or a symbolic variable
            pointing to a shared visible units bias
        """
        RBM.__init__(self, input=input, n_visible=n_visible, n_hidden=n_hidden,
                     W=W, h_bias=h_bias, v_bias=v_bias, numpy_rng=numpy_rng,
                     theano_rng=theano_rng)

    @staticmethod
    def type():
        return 'gauss-bernoulli RBM'

    def free_energy(self, v_sample):
        """
        Function to compute the free energy, it overwrite free energy function
        (here only v_bias term is different)

        :param v_sample: Sampling values of visible units
        """
        wx_b = T.dot(v_sample, self.W) + self.h_bias
        v_bias_term = 0.5 * T.dot((v_sample - self.v_bias), (v_sample - self.v_bias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - T.diagonal(v_bias_term)

    def sample_v_given_h(self, h0_sample):
        """
        This function infers state of visible units given hidden units,
        it overwrite sampling function (here you sample from normal distribution)
        :param h0_sample: Sampling values of hidden units
        """

        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''

        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        # in fact, you don't need to sample from normal distribution here and just use pre_sigmoid activation instead
        v1_sample = self.theano_rng.normal(
            size=v1_mean.shape,
            avg=v1_mean,
            std=1.0,
            dtype=theano.config.floatX) + pre_sigmoid_v1

        # v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        """
        Approximation to the reconstruction error

        :param pre_sigmoid_nv:
        """
        rms_cost = T.mean(T.sum((self.input - pre_sigmoid_nv) ** 2, axis=1))
        return rms_cost


def train_rbm(learning_rate=0.1, training_epochs=15,
              dataset=None, batch_size=20, n_visible=28 * 28,
              output_folder='gbrbm_plots', n_hidden=500, name_model='gbrbm.save'):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
|
    :param output_folder:
    :param n_hidden:
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: Theano shared array train samples and labels
    :param batch_size: size of a batch used to train the RBM
    :param name_model: Name of saved model file
    :param n_visible: Numbers of visible units
    """

    [train_set_x, train_set_y] = dataset

    # compute number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(
            numpy.zeros(
                    (batch_size, n_hidden),
                    dtype=theano.config.floatX),
            borrow=True)

    # construct the RBM class
    gbrbm = GBRBM(
        input=x,
        n_visible=n_visible,
        n_hidden=n_hidden,
        numpy_rng=rng,
        theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = gbrbm.get_cost_updates(
            lr=learning_rate,
            persistent=persistent_chain,
            k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            },
            name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=gbrbm.W.get_value(borrow=True).T,
                n_visible=n_visible,
                tile_shape=(20, 20),
                tile_spacing=(1, 1)
            )
        )

        image.save(os.path.join(output_folder, 'filters_at_epoch_%i.png' % epoch))
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    with open(os.path.join('trained_models', name_model), 'wb') as f:
        cPickle.dump(gbrbm, f, protocol=cPickle.HIGHEST_PROTOCOL)


def test_rbm(dataset=None, plot_every=1000, n_samples=10, name_model='gbrbm.save'):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    :param dataset: Theano shared array train samples and labels
    :param name_model: Name of saved model file
    :param plot_every: Number of steps before returning the sample for plotting
    :param n_samples: number of samples to plot for each chain
    """

    test_set_x, test_set_y = dataset

    with open(os.path.join('trained_models', name_model), 'rb') as f:
        gbrbm = cPickle.load(f)

    #################################
    #     Sampling from the RBM     #
    #################################
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True),
            dtype=theano.config.floatX
        )
    )

    # Print test y
    print test_set_y.eval()

    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
            gbrbm.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
            [],
            [
                hid_mfs[-1],
                hid_samples[-1],
                vis_mfs[-1],
                vis_samples[-1]
            ],
            updates=updates,
            name='sample_fn'
    )

    # find out the number of test samples
    lst_output = []
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        hid_mf, hid_sample, vis_mf, vis_sample = sample_fn()
        lst_output = hid_mf

    with open(os.path.join('trained_models', name_model), 'wb') as f:
        cPickle.dump(gbrbm, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return lst_output


if __name__ == '__main__':

    dataset ='mnist.pkl.gz'
    datasets = load_data(dataset)

    from datasets.DatasetManager import read_dataset
    train_set, valid_set, test_set = read_dataset('dataset_simulation_20.csv')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_visibles = train_set_x.shape[1]

    X_scaled = theano.shared(
        numpy.asarray(
            train_set_x,
            dtype=theano.config.floatX),
        borrow=True)

    train_rbm(dataset=(X_scaled, train_set_y),
              n_hidden=800,
              n_visible=n_visibles,
              training_epochs=50,
              batch_size=20,
              name_model='gbrbm_mnist.save',
              output_folder='gbrbm_plots')
    for i in range(0, 3, 1):
        print 'Starting i=' + str(10**i)
        test_rbm(
            dataset=(test_set_x, test_set_y),
            plot_every=10**i,
            name_model='gbrbm_mnist.save')
