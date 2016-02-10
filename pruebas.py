
import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.iscalar("A")


# Symbolic description of the result
result, updates = theano.scan(fn=lambda A: 3 * A,
                              outputs_info=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=result)

print power(1,2)
print power(1,4)

'''
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 100

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))

f = function([], T.exp(x))


print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()


t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
'''

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
          allow_input_downcast=True)

predict = theano.function(
        inputs=[x],
        outputs=prediction,
        allow_input_downcast=True)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

# theano.printing.pydotprint(f, outfile="symbolic_graph_opt.png", var_with_name_simple=True)

# import theano.d3viz as d3v
# d3v.d3viz(predict, 'mlp.html')


import theano
a = theano.tensor.vector("a")      # declare symbolic variable
b = a * a ** 5                    # build symbolic expression
f = theano.function([a], b)        # compile function
a = b
print f([0, 1, 2])
theano.printing.pydotprint(b, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)