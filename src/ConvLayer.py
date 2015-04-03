import theanets as t


class ConvLayer(t.layers.Layer):

    # Transform the inputs for this layer into an output for the layer.
    # returns:
    # output: Theano expression representing the output from this layer.
    # monitors: Outputs that can be used to monitor the state of this layer.
    # updates: A sequence of updates to apply inside a theano function.

    def transform(self, inputs):
        pass

    def setup(self):
        pass