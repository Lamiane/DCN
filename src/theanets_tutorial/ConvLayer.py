import theanets as t
import theano.tensor as TT
import architecture as arch
from scipy import signal as sgnl
import theano
import numpy

N = 10  # number of Rs
# sq_maps musza byc zawsze takie same!
sq_map_size = []  # TODO: no wlasnie? jakie?
SQ_MAPS = [] # wektor maps zadających odwzorowanie z wektora na macierz
conv_matrix = []  # TODO: uzupelnic ja czyms
# TODO: skad theanets wie, jakie parametry powinien poprawic?

# arch.create_squerization_map: moja funkcja zadająca odwzorowanie z wektora na macierz
for i in xrange(N):
    SQ_MAPS.append(arch.create_squerization_map(sq_map_size))


class ConvLayer(t.layers.Layer):

    # Transform the inputs for this layer into an output for the layer.
    # returns:
    # output: Theano expression representing the output from this layer.
    # monitors: Outputs that can be used to monitor the state of this layer.
    # updates: A sequence of updates to apply inside a theano function.

    # przystosowane z lenet_code
    def initialise(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        # (fan_in + fan_out) - normalizacja
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        # budowanie wektora wag
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # tutaj chyba zaczyna się kod, który powininen być w transform
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # convolve input feature maps with filters
        conv_out = TT.nnet.conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = sgnl.downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = TT.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

    def transform(self, inputs):
        pass

    def setup(self):
        pass