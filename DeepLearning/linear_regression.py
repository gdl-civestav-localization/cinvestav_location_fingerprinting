import scipy
import theano
import theano.tensor as T
import numpy

# Load training data.
train_x = numpy.array([[1, 9], [2, 5], [1.5, 3], [3, 7]], dtype=theano.config.floatX)
train_y = numpy.array([[2, 5], [4, 5], [3, 7], [0, 3]], dtype=theano.config.floatX)

# Parameters.
learning_rate = .001
n_steps = 10000
in_dims = 2
out_dims = 2

# Define model.
W = theano.shared(
        value=numpy.random.rand(in_dims, out_dims).astype(dtype=theano.config.floatX),
        name='w',
        borrow=True)

x = T.matrix(name='x', dtype=theano.config.floatX)
z = T.matrix(name='z', dtype=theano.config.floatX)
b = theano.shared(
        value=numpy.random.rand(out_dims, ).astype(dtype=theano.config.floatX),
        name='b',
        borrow=True
    )

# Define objective function.
y = T.dot(x, W) + b  # Linear regression.
prediction = theano.function(inputs=[x], outputs=y)  # Linear regression.

# Define loss function.
loss = T.mean((z - y) ** 2)

# Build the gradient descent algorithm.
g_W = T.grad(cost=loss, wrt=W)
g_b = T.grad(cost=loss, wrt=b)

train_model = theano.function(
    inputs=[x, z],
    outputs=loss,
    updates=[
        (W, W - learning_rate * g_W),
        (b, b - learning_rate * g_b)
    ])

# Run stupid gradient descent.
for i in range(n_steps):
    error = train_model(train_x, train_y)
    print "cost", error

print W.get_value()
print b.get_value()
print "finis"

print prediction(train_x)
