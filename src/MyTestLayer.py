import theanets
import theano.tensor as TT


class MyLayerd(theanets.layers.Layer):
    def transform(self, inputs):
        return TT.dot(inputs, self.find('w'))

    def setup(self):
        # pass
        self.log_setup(self.add_weights('w'))


layer = theanets.layers.build('mylayerd', nin=3, nout=4)

