### code based on: http://theanets.readthedocs.org/en/stable/quickstart.html

from skdata import mnist
import skdata
# from theano.tensor.TensorType import dtype
import numpy as np

# train_size = 50000
train_size = 1

def load():
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed

    def arr(n, dtype):
        # convert an array to the proper shape and dtype
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)


    train_images = arr('train_images', 'f')/255.  # regularization to [0-1]
    train_labels = arr('train_labels', np.uint8)
    test_images = arr('test_images', 'f')/255.
    test_labels = arr('test_labels', np.uint8)
    return ((train_images[:train_size], train_labels[:train_size, 0]),
            (train_images[train_size:], train_labels[train_size:, 0]),
            (test_images, test_labels[:, 0]))

    # train set: train_size
    # validation_set: 60 000 - train_size
    # test set: 10 000

