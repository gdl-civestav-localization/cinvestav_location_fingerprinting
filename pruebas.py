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
import theano.tensor as T
from theano import shared, function

# Initialization
N = 2
features = 4
x = T.matrix()
y = T.lvector()
w = shared(numpy.random.randn(features))
b = shared(numpy.zeros(()))

print "Initial model:"
print w.get_value()
print "Bias:" + str(b.get_value())

# Logistic regression model
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
# Cross-entropy loss function, negative log-likelihood
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
# Negative log-likelihood plus L2 penalty
cost = xent.mean() + 0.01 * (w ** 2).sum()

# Gradient descent of w and b
gw, gb = T.grad(cost, [w, b])
prediction = p_1 > 0.5

# Test function
predict = function(inputs=[x],
                   outputs=prediction,
                   allow_input_downcast=True)

# Train function
train = function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates={w: w - 0.1 * gw, b: b - 0.1 * gb},
        allow_input_downcast=True)


# Generate dataset
D = (numpy.random.randn(N, features),
     numpy.random.randint(size=N, low=0, high=2))

# Training process
training_steps = 10
for i in range(training_steps):
    pred, err = train(D[0], D[1])

# Final model
print "Final model:",
print w.get_value()
print "Final bias:"
print b.get_value()
print "Target values for D \t", D[1]
print D[0]
print "Prediction on D \t\t", predict(D[0])
