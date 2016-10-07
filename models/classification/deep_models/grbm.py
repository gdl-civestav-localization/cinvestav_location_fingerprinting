import copy
import os

import cPickle
import timeit

import numpy
import sys
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images, load_data
from rbm import RBM

try:
    import PIL.Image as Image
except ImportError:
    import Image
import theano


class GBRBM(RBM):
    """Gaussian Bernoulli Restricted Boltzmann Machine (RBM)"""

    def __init__(self, input=None, n_visible=784, n_hidden=500,
                 W=None, h_bias=None, v_bias=None, numpy_rng=None, theano_rng=None):
        """
        GBRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa).
        It initialize parent class (RBM).

        :param input: None for standalone RBMs or symbolic variable if RBM is part of a larger graph.

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
        RBM.__init__(
            self,
            input=input,
            n_visible=n_visible,
            n_hidden=n_hidden,
            W=W, h_bias=h_bias,
            v_bias=v_bias,
            numpy_rng=numpy_rng,
            theano_rng=theano_rng)

    @staticmethod
    def type():
        return 'gauss-bernoulli RBM'

    def __getstate__(self):
        if 'pydevd' in sys.modules:
            print 'Serializing ' + self.__class__.__name__
        state = copy.deepcopy(self.__dict__)
        del state['params']
        del state['input']
        del state['theano_rng']
        del state['L1']
        del state['L2']
        state['W'] = state['W'].get_value()
        state['h_bias'] = state['h_bias'].get_value()
        state['v_bias'] = state['v_bias'].get_value()
        return state

    def __setstate__(self, state):
        if 'pydevd' in sys.modules:
            print 'De-serializing ' + self.__class__.__name__

        numpy_rng = numpy.random.RandomState()
        rbm = GBRBM(
            input=T.matrix('x'),
            n_visible=state['n_visible'],
            n_hidden=state['n_visible'],
            W=theano.shared(value=state['W'], name='W', borrow=True),
            h_bias=theano.shared(value=state['h_bias'], name='h_bias', borrow=True),
            v_bias=theano.shared(value=state['v_bias'], name='v_bias', borrow=True),
            numpy_rng=numpy.random.RandomState(),
            theano_rng=RandomStreams(numpy_rng.randint(2 ** 30))
        )
        self.__dict__ = rbm.__dict__

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


def train_rbm(model, learning_rate=0.1, l1_learning_rate=0.001, l2_learning_rate=0.0001,training_epochs=15,
              datasets=None, batch_size=20, output_folder='rbm_plots', name_model='rbm.save'):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    :param model: Machine learning model
    :param output_folder: Output folder for weights images
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param l1_learning_rate: L1-norm's weight when added to the cost
    :param l2_learning_rate: L2-norm's weight when added to the cost
    :param datasets: Dataset with train, test and valid sets
    :param batch_size: size of a batch used to train the RBM
    :param name_model: Name of saved model file
    """
    train_set_x, train_set_y = datasets[0]

    # initialize storage for the persistent chain
    persistent_chain = theano.shared(
        numpy.zeros(
            (batch_size, model.n_hidden),
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    # get the cost and the gradient corresponding to one step of CD-k
    cost, updates = model.get_cost_updates(
        lr=learning_rate,
        persistent=persistent_chain,
        k=15,
        l1_learning_rate=l1_learning_rate,
        l2_learning_rate=l2_learning_rate
    )

    #################################
    #     Training the RBM          #
    #################################
    output_folder = os.path.join('plots', output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    index = T.lscalar()
    train_rbm = theano.function(
        [index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    for epoch in xrange(training_epochs):
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=model.W.get_value(borrow=True).T,
                n_visible=model.n_visible,
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
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)


def test_rbm(
        dataset,
        name_model,
        plot_every=1000,
        n_samples=10,
        n_chains=20
):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    :param dataset: Theano shared array train samples and labels
    :param name_model: Name of saved model file
    :param plot_every: Number of steps before returning the sample for plotting
    :param n_samples: number of samples to plot for each chain
    :param n_chains: number of parallel chain running
    """

    test_set_x, test_set_y = dataset[2]

    with open(os.path.join('trained_models', name_model), 'rb') as f:
        model = cPickle.load(f)

    #################################
    #     Sampling from the RBM     #
    #################################
    chain_input = test_set_x.get_value(borrow=True)[0:0 + n_chains]
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            chain_input,
            dtype=theano.config.floatX
        )
    )

    # define one step of Gibbs sampling
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
        model.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        inputs=[],
        outputs=[
            hid_mfs[-1],
            hid_samples[-1],
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    lst_output = []
    for idx in xrange(n_samples):
        # we discard intermediate samples because successive samples in the chain are too correlated
        hid_mf, hid_sample, vis_mf, vis_sample = sample_fn()
        print' ... plotting sample {}'.format(idx)
        print vis_mf, chain_input
        lst_output = vis_mf

    with open(os.path.join('trained_models', name_model), 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    return lst_output


if __name__ == '__main__':
    # dataset = 'mnist.pkl.gz'
    # datasets = load_data(dataset)

    from datasets.DatasetManager import read_dataset
    datasets = read_dataset('dataset_simulation_20.csv', shared=True)
    train_set_x, train_set_y = datasets[0]

    n_in = train_set_x.get_value().shape[1]
    n_out = train_set_y.get_value().shape[1]

    x = T.matrix('x')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    rbm = RBM(
        input=x,
        n_visible=n_in,
        n_hidden=1000,
        numpy_rng=rng,
        theano_rng=theano_rng
    )

    gbrbm = GBRBM(
        input=x,
        n_visible=n_in,
        n_hidden=1000,
        numpy_rng=rng,
        theano_rng=theano_rng
    )

    model = rbm

    train_rbm(
        model=model,
        datasets=datasets,
        training_epochs=15,
        batch_size=20,
        learning_rate=0.001,
        l1_learning_rate=0.001,
        l2_learning_rate=0.0001,
        name_model=model.__class__.__name__ + '_RSSI20.save',
        output_folder=model.__class__.__name__ + '_plots'
    )

    for i in range(0, 5, 1):
        print 'Starting i=' + str(10 ** i)
        test_rbm(
            dataset=datasets,
            plot_every=10 ** i,
            n_samples=1,
            n_chains=1,
            name_model=model.__class__.__name__ + '_RSSI20.save',
        )

