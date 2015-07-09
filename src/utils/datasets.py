__author__ = 'igor'
import numpy as np
import climate
logging = climate.get_logger(__name__)

try:
    import skdata.mnist
    import skdata.cifar10
except ImportError:
    logging.critical('please install skdata to run the examples!')
    raise


def load_mnist(labels=False):
    '''Load the MNIST digits dataset.'''
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed.

    def arr(n, dtype):
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)

    train_images = arr('train_images', np.float32) / 128 - 1
    train_labels = arr('train_labels', np.uint8)
    test_images = arr('test_images', np.float32) / 128 - 1
    test_labels = arr('test_labels', np.uint8)
    if labels:
        return ((train_images[:50000], train_labels[:50000, 0]),
                (train_images[50000:], train_labels[50000:, 0]),
                (test_images, test_labels[:, 0]))
    return train_images[:50000], train_images[50000:], test_images


def load_cifar(labels=False):
    cifar = skdata.cifar10.dataset.CIFAR10()
    cifar.meta  # trigger download if needed.
    pixels = cifar._pixels.astype(np.float32).reshape((len(cifar._pixels), -1)) / 128 - 1
    if labels:
        labels = cifar._labels.astype(np.uint8)
        return ((pixels[:40000], labels[:40000, 0]),
                (pixels[40000:50000], labels[40000:50000, 0]),
                (pixels[50000:], labels[50000:, 0]))
    return pixels[:40000], pixels[40000:50000], pixels[50000:]