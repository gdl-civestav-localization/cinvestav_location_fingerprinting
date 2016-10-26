""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.


Noises functions are based on

https://github.com/vitruvianscience/OpenDeep
"""

import numpy
import gzip
import os
import cPickle
import math


import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG


theano_random = RNG_MRG.MRG_RandomStreams(seed=23455)


def dropout(input, noise_level=0.5, mrg=None, rescale=True):
    """
    This is the dropout function.
    Parameters
    ----------
    input : tensor
        Tensor to apply dropout to.
    corruption_level : float
        Probability level for dropping an element (used in binomial distribution).
    mrg : random
        Random number generator with a .binomial method.
    rescale : bool
        Whether to rescale the output after dropout.
    Returns
    -------
    tensor
        Tensor with dropout applied.
    """
    if mrg is None:
        mrg = theano_random

    keep_probability = 1 - noise_level
    mask = mrg.binomial(p=keep_probability, n=1, size=input.shape, dtype=theano.config.floatX)

    output = (input * mask)

    if rescale:
        output /= keep_probability

    return output


def add_gaussian(input, noise_level=1, mrg=None):
    """
    This takes an input tensor and adds Gaussian noise to its elements with mean zero and provided standard deviation.
    Parameters
    ----------
    input : tensor
        Tensor to add Gaussian noise to.
    noise_level : float
        Standard deviation to use.
    mrg : random
        Random number generator with a .normal method.
    Returns
    -------
    tensor
        Tensor with Gaussian noise added.
    """
    if mrg is None:
        mrg = theano_random

    noise = mrg.normal(avg=0, std=noise_level, size=input.shape, dtype=theano.config.floatX)
    output = input + noise
    return output


def add_uniform(input, noise_level, mrg=None):
    """
    This takes an intput tensor and adds noise drawn from a Uniform distribution from +- interval.
    Parameters
    ----------
    input : tensor
        Tensor to add uniform noise to.
    noise_level : float
        Range for noise to be drawn from (+- interval).
    mrg : random
        Random number generator with a .uniform method.
    Returns
    -------
    tensor
        Tensor with uniform noise added.
    """
    if mrg is None:
        mrg = theano_random

    noise = mrg.uniform(low=-noise_level, high=noise_level, size=input.shape, dtype=theano.config.floatX)
    output = input + noise
    return output


def salt_and_pepper(input, noise_level=0.2, mrg=None):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.
    Parameters
    ----------
    input : tensor
        The tensor to apply salt and pepper noise to.
    noise_level : float
        The amount of salt and pepper noise to add.
    mrg : random
        Random number generator with .binomial method.
    Returns
    -------
    tensor
        Tensor with salt and pepper noise applied.
    """
    if mrg is None:
        mrg = theano_random

    a = mrg.binomial(size=input.shape, n=1, p=(1 - noise_level), dtype=theano.config.floatX)
    b = mrg.binomial(size=input.shape, n=1, p=0.5, dtype=theano.config.floatX)
    c = T.eq(a, 0) * b
    return input * a + c


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1
    :param eps:
    :param ndar:
    """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def zero_mean_unit_variance(Data):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    Mean = numpy.mean(Data, axis=0)
    Data -= Mean

    Std = numpy.std(Data, axis=0)
    index = (numpy.abs(Std < 10 ** -5))
    Std[index] = 1
    Data /= Std
    return [Data, Mean, Std]


def normalize(Data, Mean, Std):
    Data -= Mean
    Data /= Std
    return Data


def max_divisor(a):
    b = int(round(math.sqrt(a)))
    for i in range(b, 0, -1):
        res = a % i
        if res == 0:
            return i
    return 1


def tile_raster_images(X, n_visible, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing d whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :param tile_spacing:

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type n_visible: Int;
    :param n_visible: Number of visibles units

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    comun_divisor = max_divisor(n_visible)
    img_shape = (comun_divisor, n_visible / comun_divisor)

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def load_data(dataset):
    """ Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "datasets",
            "dataset",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        :param data_xy:
        :param borrow:
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(
            numpy.asarray(
                data_x,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )

        shared_y = theano.shared(
            numpy.asarray(
                data_y.reshape((len(data_y), 1)),
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y  # T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
