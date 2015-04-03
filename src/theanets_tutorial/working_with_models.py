import theanets as t
import theano.tensor as TT
import numpy as np

netA = t.Autoencoder()
netR = t.Regressor()
netC = t.Classifier(layers=(4, 5, 6, 2))  # outputs a probability distribution over available labels
netD = t.Classifier(layers=(4, (5, "relu"), 6, 2))  # second layer has rectified linear activation
                                                    # and is a feed-forward layer (default)
netE = t.recurrent.Classifier(layers=(4, (5, "rnn"), 6, 2))  # netE is a recurrent classifier and second layer
                                                             # is recurrent with sigmoid activation (default)
# netF's second layer has 5 nodes with tanh activation function, it's recurrent and has dropout 0.3
netF = t.Regressor(layers=(4, dict(size=5, activation='tanh', form="rnn", dropout="0.3"), 2))

# For activation functions parameters see:
# http://theanets.readthedocs.org/en/stable/creating.html#activation-functions

# regularization parameter is equal to 0.1
exp = t.Experiment(t.Classifier, layers=(784, 1000, 784), hidden_l1=0.1)


# Customizing layers
class MyLayer(t.layers.Layer):
    def transform(self, inputs):
        return TT.dot(inputs, self.find('w'))

    def setup(self):
        self.log_setup(self.add_weights('w'))

# how to use MyLayer:
layer = t.layers.build('mylayer', nin=3, nout=4)
# or
netML = t.Autoencoder(layers=(4, ('mylayer', 'linear', 3), 4), tied_weights=True)
# nin - size of input vectors to this layer
# nout - size of output vectors produced by this layer
# droput - set the given fraction of outputs in this layer randomly to 0
# log_setup(count) - log some information about this layer, count - number of parameter values in this layer
# setup() - set up the parameters and initial values for this layer
# add_weights() - helper method to create a new weight matrix
# find() - get a shared variable for a parameter by name
# transform(input) - transform the inputs for this layer into an output for the layer

# TRAINING A MODEL
training_data = []
validation_data = []
exp.train(training_data, validation_data, optimize='nag', learning_rate=0.01, momentum=0.9)

# optimize - optimization algorithm used
# nag - Nesterov's accelerated gradient
# sgd - stochastic gradient descent
# rprop - resilent backpropagation
# rmsprop
# and others as well, see:
# http://theanets.readthedocs.org/en/stable/training.html#gradient-based-methods

# PROVIDING DATA
dataset = np.load("file")


# TRAINING
for train, valid in exp.itertrain(training_data, validation_data, **kwargs):
    print('training loss:', train['loss'])
    print('most recent validation loss:', valid['loss'])

# The Experiment class can snapshot your model automatically during training.
# When you call Experiment.train(), you can provide the following keyword arguments:
# save_progress: This should be a string containing a filename where the model should be saved.
# You can also save and load models manually by calling Experiment.save() and Experiment.load(), respectively.

# PREDICTING
new_dataset = []
results = exp.network.predict(new_dataset)
classes = exp.network.classify(new_dataset)  # for classifiers

# Network.find()
# The parameters in each layer of the model are available using Network.find()
values = netA.find(1, 0).get_value()