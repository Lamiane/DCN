__author__ = 'agnieszka'
import sys
sys.path.append('..')
from no_threshold import F1Score
from get_predictions import Predictor


# TODO testing!
class TwoThresholdWRTF1Score(F1Score):
    def __init__(self, save_best_model_path=None):
        super(TwoThresholdWRTF1Score, self).__init__()
        self.thresholds_list = []
        self.saving_path = save_best_model_path

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

        self.thresholds_list.append(threshold)
        self.score_list.append(score)

        if self.saving_path is not None:
            if max(self.score_list) == score:
                pass
                # TODO: saving here

        print "TwoThreshold score", score, "\ncorresponding threshold pair:", threshold

    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        # F1_score = 2TP/(2TP + FN + FP)
        # our F1_score to deal with not classifying some examples:
        # if some example isn't classified (is between thresholds) it counts as 1/2 of classifying it wrongly
        # therefore the F1_score becomes: 2TP/(2TP + FN +FP + 0.5 * unclassified)

        from numpy import argmax, mean
        import sys
        sys.path.append('..')
        from utils import values

        dic = {}
        # the threshold will be symmetric, so we only care about how far away is the prediction from 0.5
        # therefore we fold the dictionary in half
        for index in xrange(len(true_y)):
            if argmax(predictions[index]) != true_y[index]:     # FALSE NEGATIVE OR FALSE POSITIVE
                # floating to get a hashable float instead of unhashable numpy array
                # we also need max max because numpy...
                # dic[float(abs(0.5 - max(max(predictions[index]))))] = values.FN_FP
                dic[float(max(max(predictions[index])))] = values.FN_FP
            elif argmax(predictions[index]) == 1:   # TRUE POSITIVE
                # dic[float(abs(0.5 - max(max(predictions[index]))))] = values.TP
                dic[float(max(max(predictions[index])))] = values.TP
            # else TRUE NEGATIVE, we have no interest in this one

        TP = sum(1 for x in dic.values() if x == values.TP)
        FP_FN = len(dic)-TP
        unclassified = 0
        scores_down = []
        scores_up = []
        sorted_dic_keys = sorted(dic)   # don't want to sort dic over and over again

        # down threshold
        for key in sorted_dic_keys:
            if key > 0.5:
                break
            unclassified += 1
            if dic[key] == values.FN_FP:   # it was FN or FP
                FP_FN -= 1
            else:
                TP -= 1         # it was TP
            scores_down.append(2*TP/(2*TP + FP_FN + 0.5 * unclassified))

        for key in sorted_dic_keys:
            if key < 0.5:
                continue
            unclassified += 1
            if dic[key] == values.FN_FP:   # it was FN or FP
                FP_FN -= 1
            else:
                TP -= 1         # it was TP
            scores_up.append(2*TP/(2*TP + FP_FN + 0.5 * unclassified))

        best_score_index_down = argmax(scores_down)
        best_score_index_up = argmax(scores_up)
        # TODO: max score musi zostac jakos policzone!
        max_score = scores[best_score_index]

        # down
        t1_down = sorted_dic_keys[best_score_index_down]
        # TODO: minus czy plus?
        t2_down = sorted_dic_keys[best_score_index_down-1]
        best_threshold_down = mean([t1_down, t2_down])

        # up
        t1_up = sorted_dic_keys[best_score_index_up]
        # TODO: minus czy plus?
        t2_up = sorted_dic_keys[best_score_index_up-1]
        best_threshold_up = 0.5 + mean([t1_up, t2_up])

        return (best_threshold_down, best_threshold_up), max_score
