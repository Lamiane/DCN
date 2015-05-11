import theanets
import theano.tensor as TT


class MyLayerd(theanets.layers.Layer):
    def transform(self, inputs):
        return TT.dot(inputs, self.find('w'))

    # def setup(self):
    #     pass
    #     # self.log_setup(self.add_weights('w'))
    #

layer = theanets.layers.build('mylayerd', nin=3, nout=4)

net = theanets.Autoencoder(
    layers=(4, ('mylayerd', 'linear', 3), 4),
    tied_weights=False,
)