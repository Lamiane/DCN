__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
import numpy as np
from sklearn.metrics import f1_score
import sys
sys.path.append('..')
import configuration.model as config
from get_predictions import Predictor


class F1Score(TrainExtension):
    def __init__(self):
        self.predictor = None
        self.score_list = []

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm):
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        # obtaining validating set # TODO: finally we want to have train-validation-test set. Or sth.
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y

        # TODO: this FOR should be in get_predictions.Predictor.get_prediction function
        y_pred = []
        for i in xrange(len(valid_x)):
            # TODO: perhaps config.data_height and config.data_width are obtainable from parameters?
            # TODO: perhaps input space is obtainale from model?
            sample = np.reshape(valid_x[i], (1, config.data_height, config.data_width, 1))
            y_pred.append(np.argmax(self.predictor.get_prediction(sample)))

        score = f1_score(y_true=valid_y, y_pred=y_pred)
        self.score_list.append(score)

    # we don't need it, let's inherit the default empty method
    # def on_save(self, model, dataset, algorithm):
    #     pass


class F1Score1Threshold(F1Score):
    def __init__(self):
        super(F1Score1Threshold, self).__init__()
        # self.predictor = None   # TODO: wywalic
        # self.score_list = []    # TODO: wywalic
        self.threshold_list = []

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm, t=0.5):
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        # obtaining validating set # TODO: finally we want to have train-validation-test set. Or sth.
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y

        # TODO: this FOR should be in get_predictions.Predictor.get_prediction function
        y_pred = []
        for i in xrange(len(valid_x)):
            # TODO: perhaps config.data_height and config.data_width are obtainable from parameters?
            # TODO: perhaps input space is obtainale from model?
            sample = np.reshape(valid_x[i], (1, config.data_height, config.data_width, 1))
            predicted_outcome = self.predictor.get_prediction(sample)
            if predicted_outcome[1] > t:
                y_pred.append(1)
            else:
                y_pred.append(0)

        score = f1_score(y_true=valid_y, y_pred=y_pred)
        self.threshold_list.append(t)
        self.score_list.append(score)

    # we don't need it, let's inherit the default empty method
    # def on_save(self, model, dataset, algorithm):
    #     pass