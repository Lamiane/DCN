__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
import numpy as np
from sklearn.metrics import f1_score
import sys
sys.path.append('..')
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

        y_pred = self.predictor.get_predictions(valid_x)
        y_classes = [np.argmax(pred) for pred in y_pred]
        score = f1_score(y_true=valid_y, y_pred=y_classes)
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

    def on_monitor(self, model, dataset, algorithm):
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        # obtaining validating set # TODO: finally we want to have train-validation-test set. Or sth.
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y

        predictions = self.predictor.get_predictions(valid_x)
        threshold, score = self.compute_optimal_threshold_and_score(valid_y, predictions)

        self.threshold_list.append(threshold)
        self.score_list.append(score)

    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        # F1_score = 2TP/(2TP + FN + FP)
        # our F1_score to deal with not classifying some examples:
        # if some example isn't classified (is between thresholds) it counts as 1/2 of classifying it wrongly
        # therefore the F1_score becomes: 2TP/(2TP + FN +FP + 0.5 * unclassified)

        from numpy import argmax, mean

        dic = {}
        # the threshold will be symmetric, so we only care about how far away is the prediction from 0.5
        # therefore we fold the dictionary in half
        for index in xrange(len(true_y)):
            if argmax(predictions[index]) != true_y[index]:     # FALSE NEGATIVE OR FALSE POSITIVE
                dic[abs(0.5 - max(predictions[index]))] = 'FN_FP'
            elif argmax(predictions[index]) == 1:   # TRUE POSITIVE
                dic[abs(0.5 - max(predictions[index]))] = 'TP'
            # else TRUE NEGATIVE, we have no interest in this one

        TP = sum(dic.values())
        FP_FN = len(dic)-TP
        skipped = 0
        scores = []
        for key in sorted(dic):
            skipped += 1
            if dic[key] == 'FN_FP':   # it was FN or FP
                FP_FN -= 1
            else:
                TP -= 1         # it was TP
            scores.append(2*TP/(2*TP + FP_FN + 0.5 * skipped))

        max_score = max(scores)
        best_threshold = mean(sorted(dic)[argmax(scores)], sorted(dic)[argmax(scores)-1])

        return best_threshold, max_score



        # we don't need it, let's inherit the default empty method
        # def on_save(self, model, dataset, algorithm):
        #     pass