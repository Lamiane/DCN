__author__ = 'nex'
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial
import sys
sys.path.append('..')
from no_threshold import F1Score
from get_predictions import Predictor


class ROC_Yoduen(F1Score):
    def __init__(self, save_best_model_path=None, save=False):
        super(ROC_Yoduen, self).__init__()
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
        best_threshold, best_score = self.compute_optimal_threshold_and_score(valid_y, predictions)

        self.thresholds_list.append(best_threshold)
        self.score_list.append(best_score)

        if self.saving_path is not None and self.save:
            if max(self.score_list) == best_score:
                try:
                    # Make sure that saving does not serialize the dataset
                    dataset._serialization_guard = SerializationGuard()
                    save_path = self.saving_path
                    serial.save(save_path, model,
                                on_overwrite='backup')
                finally:
                    dataset._serialization_guard = None

        print "\n\nROC using Youden metric\n score:", best_score, "\ncorresponding threshold:", best_threshold

    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        axis = sorted([[pred[0][1], label] for pred, label in zip(predictions, true_y)])

        # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
        # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
        # middle    -1    [[ 0. 0. 1. ]]

        actives = sum([1 for arr_pred_lab in axis if arr_pred_lab[1] == 1])
        nonactives = len(axis) - actives

        # threshold is zero, it means we label ALL the samples as negatives
        TP = 0
        FP = 0
        TN = nonactives
        FN = actives

        best_score = 0
        best_threshold = 0
        next_pred_after_best_threshold = 0
        update_next = False
        for prediction, label in axis:
            if label == 1:
                TP += 1     # after moving threshold we have one well classified positive example more
                FN -= 1     # and that means we classify wrongly one positive example less
            else:   # label is 0 - means nonactive
                FP -= 1     # we have one wrongly classified negative example more
                TN += 1     # so that's one well classified negative example less
            # calculating score according to Youden's metric
            print 'TP:', TP, '\tFP:', FP, '\nFN', FN, '\tTN', TN
            score = (float(TP)/(float(TP) + FN)) - (float(FP)/(float(FP) + TN))
            if update_next:
                update_next = False
                next_pred_after_best_threshold = prediction
            if score > best_score:
                best_score = score
                best_threshold = prediction
                update_next = True

        # compute optimal threshold
        best_threshold = (best_threshold + next_pred_after_best_threshold)/2.0
        return best_threshold, best_score
