from theano import shared, function
from numpy import array, size, zeros
import architecture as A
from theano.tensor import dmatrix
from theano.tensor.signal import downsample


def main():
    vec = array([[1, 2, 3], [4, 5, 6]])
    input_f = dmatrix('input')
    max_pool_shape = (2, 2)
    pool_out = downsample.max_pool_2d(input_f, max_pool_shape, ignore_border=False)
    f = function([input_f], pool_out)
    print vec
    print f(vec)