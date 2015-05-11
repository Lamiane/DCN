### code based on: http://theanets.readthedocs.org/en/stable/quickstart.html
from time import time
import theanets as t
from mnist_data import load as load_mnist
import numpy as np
import notificator
import cPickle as pickle
import matplotlib.pyplot as plt
import theano.tensor as TT


class MyLayerd(t.layers.Layer):
    def transform(self, inputs):
        return TT.dot(inputs, self.find('w'))

    def setup(self):
        # pass
        self.log_setup(self.add_weights('w'))


start_time = time()

# defining the model architecture
exp = t.Experiment(t.Classifier, layers=(784, 100, ('mylayerd', 'linear', 3), 10))

# theanets initializes the parameters of a model randomly

# preparing the data
train, valid, test = load_mnist()

prep_time = time()

# training
exp.train(train, valid, optimize='nag', learning_rate=1e-3, momentum=0.9)
# if validation set is not provided train set will be used for this purpose
# optimize = an algorithm to use for training, RmsProp is the default
# nag: Nestrov's Accelerated Gradient:
# http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.NAG.html#theanets.trainer.NAG

train_time = time()

# the code below doesn't work due to some QT weirdos
# visualization
#img = np.zeros((28*10, 28*10), dtype='f')
#for i, pix in enumerate(exp.network.find(1, 0).get_value().T):
#    r, c = divmod(i, 10)
#    img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()

counter_good = 0
counter_bad = 0
# bad_details = []

for i, element in enumerate(test[0]):
    element = np.reshape(element, (1, 784))
    predicted_class = exp.network.predict(element)
    predicted_tag = np.argmax(predicted_class)  # np.argmax() returns the index of a max value
    true_tag = test[1][i]
    v = predicted_tag == true_tag
    if v.all():
        counter_good += 1
    else:
        counter_bad += 1
        # bad_details.append([predicted_class, test[1][i]])

# print bad_details

all_time = time() - start_time
vis_time = time() - train_time
train_time = train_time - prep_time
prep_time = prep_time - start_time

print "all:", all_time/60.0
print "prep:", prep_time/60.0
print "train:", train_time/60.0
print "vis:", vis_time/60.0

print "good:", counter_good
print "bad:", counter_bad
print "all:", counter_bad+counter_good
# print "bad details:", bad_details

# not needed anymore
# with open("baddies.p", "w") as FILE_BAD:
#     pickle.dump(bad_details, FILE_BAD)

# pickling trained network for later use
exp.save("network.p")

# notifying the program is finished
notificator.notify()
