import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.nonlinearities import softmax, sigmoid, tanh, rectify
from lasagne.updates import nesterov_momentum
from lasagne.regularization import l1, l2, regularize_network_params
# configuring the network
l_in = InputLayer((20844, 1))   # or the other way around?
h1 = DenseLayer(l_in, num_units=100, nonlinearity=sigmoid)
h2 = DenseLayer(h1, num_units=100, nonlinearity=sigmoid)
out = DenseLayer(h2, num_units=2, nonlinearity=softmax)

# cross validation

# reading in the data
x = None    # must be theano vector
y = None    # must be theano vector

# actually learning
l_out = get_output(out, x)
params = lasagne.layers.get_all_params(out)
loss = T.mean(T.nnet.categorical_crossentropy(l_out, y))
regularization = regularize_network_params(out, penalty=l1)
updates_nestrov = nesterov_momentum(loss, params, momentum=0.9, learning_rate=0.1)
train_function = theano.function([x, y], updates=updates_nestrov)
loss = loss + regularization
