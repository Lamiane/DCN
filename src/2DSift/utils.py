import climate
import pickle
import gzip
import numpy as np
import os
import tempfile
import itertools
from copy import copy

logging = climate.get_logger(__name__)

climate.enable_default_logging()

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.critical('please install matplotlib to run the examples!')
    raise

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


def plot_images(imgs, loc, title=None, channels=1):
    '''Plot an array of images.

    We assume that we are given a matrix of data whose shape is (n*n, s*s*c) --
    that is, there are n^2 images along the first axis of the array, and each
    image is c squares measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0]) / channels))
    assert s * s == len(imgs[0]) / channels, 'images must be square!'

    img = np.zeros((s * n, s * n, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * s:(r + 1) * s, c * s:(c + 1) * s] = pix.reshape((s, s, channels))

    img -= img.min()
    img /= img.max()

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
    if title:
        ax.set_title(title)


def plot_layers(weights, tied_weights=False, channels=1):
    '''Create a plot of weights, visualized as "bottom-level" pixel arrays.'''
    if hasattr(weights[0], 'get_value'):
        weights = [w.get_value() for w in weights]
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + i + 1,
                    channels=channels,
                    title='Layer {}'.format(i + 1))
    weight = weights[-1]
    n = weight.shape[1] / channels
    if int(np.sqrt(n)) ** 2 != n:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Layer {}'.format(k))
    else:
        plot_images(weight,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Decoding weights')


def prime_factors(n):
    """Finds the prime factors of 'n'"""
    from math import sqrt

    pFact, limit, check, num = [], int(sqrt(n)) + 1, 2, n
    if n == 1: return [1]
    for check in range(2, limit):
        while num % check == 0:
            pFact.append(check)
            num /= check
    if num > 1:
        pFact.append(num)
    return pFact


def bufcount(filenames=[]):
    lines = []
    for filename in filenames:
        f = open(filename)
        buf_size = 1024 * 1024
        read_f = f.read  # loop optimization
        this_file_lines = 0
        buf = read_f(buf_size)
        while buf:
            this_file_lines += buf.count('\n')
            buf = read_f(buf_size)
        lines.append(this_file_lines)

    return lines


def splits(examples, cv):
    """

    :rtype : object
    """

    def compute_split(exmpls, cvno):
        local_split = [int(np.ceil((exmpls / float(cvno)) * i)) for i in range(cvno)]
        spl = list(local_split)
        spl.append(exmpls)
        lns = [spl[i + 1] - spl[i] for i in range(len(spl) - 1)]
        if max(lns) > min(lns):
            lns.sort(reverse=True)
            local_split = [0]
            for i in lns:
                local_split.append(local_split[-1] + i)
            local_split = local_split[:-1]
        return local_split

    def is_power_of_2(n):
        if (n > 0 and (n & (n - 1))) == 0:
            return True
        else:
            return False

    split = compute_split(exmpls=examples, cvno=cv)
    # compute the best batch size and if some examples should be removed/added
    # first column (== row index): batch size, second: how many examples to remove, third: how many to add
    min_bs, max_bs = 8, 16
    rtab = np.zeros((max_bs - min_bs + 1, 3), dtype=int)
    for k, bs in enumerate(range(min_bs, max_bs + 1)):
        rtab[k, :] = bs, examples % (cv * bs), ((cv * bs) - (examples % (cv * bs))) % (cv * bs)
    # choose the best solution
    # sort the array by the last column (number of elements to add)
    # rtab = rtab[min_bs:max_bs + 1]
    rtab = rtab[rtab[:, 2].argsort()]
    # TODO select a best method to choose best
    # prefer even numbers, prefer powers of 2, prefer one that nothing is to be changed
    # first choose parameters for the REMOVE option
    rtab = rtab[rtab[:, 1].argsort()]
    indx = np.where(rtab[:, 1] == rtab[0, 1])
    if len(indx[0]) == 1:
        # only one row with smallest number of items to add
        new_examples = examples - rtab[0, 1]
        remove_ans = [rtab[0, 0], new_examples, compute_split(exmpls=new_examples, cvno=cv)]
    else:
        # if more answers with smallest number of items to remove are available, choose the best
        # limit rtab to rows with minimal number of examples to remove
        rem_rt = rtab[indx[0], :]
        # first check for a power of 2 in the batch_size column (rtab[indx, 1])
        l = [i for i in range(rem_rt.shape[0]) if is_power_of_2(n=rem_rt[i, 0])]
        if len(l) > 0:
            # at least one power of 2 found, select the first
            p2_ind = l[0]
            new_examples = examples - rem_rt[l[0], 1]
            remove_ans = [rem_rt[l[0], 0], new_examples, compute_split(new_examples, cv)]
        else:
            # check if any of solutions is a multiple of 2
            l = [i for i in range(rem_rt.shape[0]) if rem_rt[i, 0] % 2 == 0]
            if len(l) > 0:
                # at least one multiple of 2 found
                new_examples = examples - rem_rt[l[0], 1]
                remove_ans = [rem_rt[l[0], 0], new_examples, compute_split(new_examples, cv)]
            else:
                # all solutions are odd, choose the first one
                new_examples = examples - rtab[0, 1]
                remove_ans = [rtab[0, 0], new_examples, compute_split(exmpls=new_examples, cvno=cv)]
    # perform the same computations for the ADD EXAMPLES option
    rtab = rtab[rtab[:, 2].argsort()]
    indx = np.where(rtab[:, 2] == rtab[0, 2])
    if len(indx[0]) == 1:
        new_examples = examples + rtab[0, 2]
        add_ans = [rtab[0, 0], new_examples, compute_split(exmpls=new_examples, cvno=cv)]
    else:
        add_rt = rtab[indx[0], :]
        l = [i for i in range(add_rt.shape[0]) if is_power_of_2(n=add_rt[i, 0])]
        if len(l) > 0:
            # at least one power of 2 found, select the first
            p2_ind = l[0]
            new_examples = examples + add_rt[l[0], 2]
            add_ans = [add_rt[l[0], 0], new_examples, compute_split(new_examples, cv)]
        else:
            l = [i for i in range(add_rt.shape[0]) if add_rt[i, 0] % 2 == 0]
            if len(l) > 0:
                # at least one multiple of 2 found
                new_examples = examples + add_rt[l[0], 2]
                add_ans = [add_rt[l[0], 0], new_examples, compute_split(new_examples, cv)]
            else:
                # all solutions are odd, choose the first one
                new_examples = examples + rtab[0, 2]
                add_ans = [rtab[0, 0], new_examples, compute_split(exmpls=new_examples, cvno=cv)]
    # draw a list of examples to be added in case they are not shuffled, otherwise take those from the end
    # TODO rewrite so that drawing is always identical (constant seed? constant seed depending on original set?)
    add_ans.append(list(np.random.randint(examples, size=add_ans[1]-examples)))
    return split, remove_ans, add_ans


def common(pfa, pfb):
    c = []
    tmpb = copy(pfb)
    for i in pfa:
        if i in tmpb:
            c.append(i)
            for k in range(len(tmpb)):
                if tmpb[k] == i:
                    if k == 0:
                        tmpb = tmpb[1:]
                    elif k == len(tmpb):
                        tmpb = tmpb[:-1]
                    else:
                        ntmpb = tmpb[:k]
                        ntmpb.extend(tmpb[k + 1:])
                        tmpb = ntmpb
                    break
    return c


def combinations(common):
    for s in xrange(0, len(common) + 1):
        for comb in itertools.combinations(common, s):
            yield comb

