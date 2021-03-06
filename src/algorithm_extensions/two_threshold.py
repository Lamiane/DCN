__author__ = 'agnieszka'
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial
import sys
sys.path.append('..')
from no_threshold import F1Score
from get_predictions import Predictor


class TwoThresholdWRTF1Score(F1Score):
    def __init__(self, save_best_model_path=None, save=False):
        super(TwoThresholdWRTF1Score, self).__init__()
        self.thresholds_list = []
        self.saving_path = save_best_model_path
        self.save = save

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
        lower_threshold, upper_threshold, score = self.compute_optimal_threshold_and_score(valid_y, predictions)

        self.thresholds_list.append((upper_threshold, lower_threshold))
        self.score_list.append(score)

        if self.saving_path is not None and self.save:
            if max(self.score_list) == score:
                try:
                    # Make sure that saving does not serialize the dataset
                    dataset._serialization_guard = SerializationGuard()
                    save_path = self.saving_path
                    serial.save(save_path, model,
                                on_overwrite='backup')
                finally:
                    dataset._serialization_guard = None

        print "\n\nTwoThreshold score", score, "\ncorresponding threshold pair:", lower_threshold, ':', upper_threshold

    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        axis = sorted([[pred[0][1], label] for pred, label in zip(predictions, true_y)])

        x = len(axis)
        y = 0

        while y < x and axis[y][1] == 0:
            y += 1

        while x > y and axis[x-1][1] == 1:
            x -= 1

        tn_global = y
        tp_global = len(axis) - x
        fn_global = 0
        fp_global = 0

        maximal_score = -1.0
        best_y_idx = y
        best_x_idx = x

        for i in xrange(y, x+1):
            tp_local = tp_global
            fp_local = fp_global

            for j in xrange(x, i-1, -1):
                middle = len(axis) - tp_local - fp_local - tn_global - fn_global
                if tp_local == 0:
                    score_local = 0
                else:
                    score_local = 2.0 * tp_local / (2 * tp_local + fp_local + fn_global + 0.5 * middle)

                if score_local > maximal_score:
                    maximal_score = score_local
                    best_y_idx, best_x_idx = i, j

                if axis[j-1][1] == 1:
                    tp_local += 1
                else:
                    fp_local += 1

            if i == x:
                break

            if axis[i][1] == 0:
                tn_global += 1
            else:
                fn_global += 1

        best_y = 0 if best_y_idx == 0 else 1
        if len(axis) > best_y_idx > 0:
            best_y = (axis[best_y_idx-1][0] + axis[best_y_idx][0]) / 2.0

        best_x = 0 if best_x_idx == 0 else 1
        if len(axis) > best_x_idx > 0:
            best_x = (axis[best_x_idx-1][0] + axis[best_x_idx][0]) / 2.0

        return best_y, best_x, maximal_score
