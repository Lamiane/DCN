# This code is based on code from: http://fastml.com/how-to-get-predictions-from-pylearn2/
# from pylearn2.utils import serial
# from theano import tensor as T
import theano


class Predictor(object):

    def __init__(self, model):
        # model: a model, such like those models from pikles
        # model = serial.load(model_path)     # some pkl
        self.X = model.get_input_space().make_theano_batch()
        self.Y = model.fprop(self.X)
        # I return raw values, and take argmax in F1Score
        # self.Y = T.argmax(self.Y, axis=1)
        self.f = theano.function([self.X], self.Y)

        in_space = model.get_input_space()
        if in_space.axes == ('c', 0, 1, 'b'):
            self.input_space = (in_space.num_channels, in_space.shape[0], in_space.shape[1], 1)
        elif in_space.axes == ('b', 0, 1, 'c'):
            self.input_space = (1, in_space.shape[0], in_space.shape[1], in_space.num_channels)
        else:
            raise ValueError("Model has unsupported input space. "
                             "Supported types are: Conv2Space, axes = ('c', 0, 1, 'b') or ('b', 0, 1, 'c')"
                             "Batch size must be one.")

    def get_prediction(self, x_test):
        y = self.f(x_test)
        return y

    def get_predictions(self, x_vec):
        import numpy as np
        pred_vec = []

        for index in xrange(len(x_vec)):
            sample = np.reshape(x_vec[index], self.input_space)
            y = self.f(sample)
            pred_vec.append(y)

        return pred_vec

