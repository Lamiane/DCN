from pylearn2.utils import serial
from theano import tensor as T
import theano


class Predictor(object):

    def __init__(self, model_path):
        model = serial.load(model_path) # some pkl
        self.X = model.get_input_space().make_theano_batch()
        self.Y = model.fprop(self.X)
        self.Y = T.argmax(self.Y, axis=1)
        self.f = theano.function([self.X], self.Y)

    def return_prediction(self, x_test):
        y = self.f(x_test)
        return y