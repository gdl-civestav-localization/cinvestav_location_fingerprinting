from theano import function, config, shared, tensor, sandbox
import numpy
import time
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

__author__ = 'Gibran Felix'


def test_gpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], tensor.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')


if __name__ == '__main__':
    test_gpu()