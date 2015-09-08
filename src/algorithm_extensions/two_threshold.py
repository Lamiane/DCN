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
        upper_threshold, lower_threshold, score = self.compute_optimal_threshold_and_score(valid_y, predictions)

        self.thresholds_list.append((upper_threshold, lower_threshold))
        self.score_list.append(score)

        if self.saving_path is not None:
            if max(self.score_list) == score:
                pass
                # TODO: saving here

        print "TwoThreshold score", score, "\ncorresponding threshold pair:", lower_threshold, ':', upper_threshold

    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        # F1_score = 2TP/(2TP + FN + FP)
        # our F1_score to deal with not classifying some examples:
        # if some example isn't classified (is between thresholds) it counts as 1/2 of classifying it wrongly
        # therefore the F1_score becomes: 2TP/(2TP + FN +FP + 0.5 * unclassified)
        from numpy import argmax, mean
        from utils import values
        import sys

        # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
        # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
        # middle    -1    [[ 0. 0. 1. ]]

        dic = {}
        for pred, tr_y in zip(predictions, true_y):
            dic[pred[0][1]] = tr_y

        # calculating down threshold influence
        sorted_dic_keys = sorted(dic)   # sorting from lowest to highest
        print 'sorted dic keys', sorted_dic_keys    # TODO: check and remove in the future
        score = len(sorted_dic_keys)*0.5    # they're all middle for now
        min_score = sys.maxint
        minimums = {}
        current_minimum_key = 'this value should be overwritten before entering the dictionary'
        mid_fn_dic = {}
        num_middle = len(sorted_dic_keys)
        num_fn = 0
        # mid_fn_dic[0] = (num_middle, num_fn)
        for key in sorted_dic_keys:
            score -= 0.5        # another sample is classified so one middle less
            num_middle -= 1
            if dic[key] == 1:   # if it was active, then it will be classified wrongly (false negative)
                score += 1
                num_fn += 1
            if score <= min_score:
                min_score = score
                current_minimum_key = key
            minimums[key] = current_minimum_key     # storing key on which we had minimum
            mid_fn_dic[key] = (num_middle, num_fn)

        num_tp = 0
        num_fp = 0
        up_scores = {}
        for key in reversed(sorted_dic_keys):
            if dic[key] == 1:   # it was active, we will classify it as active, num_tp grows
                num_tp += 1
            else:
                num_fp += 1
            best_down_threshold_key = minimums[key]
            best_down_threshold_num_mid, best_down_threshold_num_fn = mid_fn_dic[best_down_threshold_key]
            num_middle = best_down_threshold_num_mid - num_tp - num_fp  # some examples are now classified
            up_score = (2 * float(num_tp))/(2 * float(num_tp) + num_fp + num_fn + 0.5 * float(num_middle))
            up_scores[key] = up_score

        key_with_maximal_up_threshold_score = max(up_scores, key=up_scores.get)
        respective_down_threshold = minimums[key_with_maximal_up_threshold_score] + sys.float_info.min
        key_with_maximal_up_threshold_score -= sys.float_info.min
        maximal_score = max(up_scores, key=up_scores.get)

        return key_with_maximal_up_threshold_score, respective_down_threshold, maximal_score





    # TP = sum(1 for x in dic.values() if x == values.TP)
    # FP_FN = len(dic)-TP
    # unclassified = 0
    # scores_down = []
    # scores_up = []
    # sorted_dic_keys = sorted(dic)   # don't want to sort dic over and over again
    #
    # # down threshold
    # for key in sorted_dic_keys:
    #     if key > 0.5:
    #         break
    #     unclassified += 1
    #     if dic[key] == values.FN_FP:   # it was FN or FP
    #         FP_FN -= 1
    #     else:
    #         TP -= 1         # it was TP
    #     scores_down.append(2*TP/(2*TP + FP_FN + 0.5 * unclassified))
    #
    # for key in sorted_dic_keys:
    #     if key < 0.5:
    #         continue
    #     unclassified += 1
    #     if dic[key] == values.FN_FP:   # it was FN or FP
    #         FP_FN -= 1
    #     else:
    #         TP -= 1         # it was TP
    #     scores_up.append(2*TP/(2*TP + FP_FN + 0.5 * unclassified))
    #
    # best_score_index_down = argmax(scores_down)
    # best_score_index_up = argmax(scores_up)
    # # TODO: max score musi zostac jakos policzone!
    # max_score = scores[best_score_index]
    #
    # # down
    # t1_down = sorted_dic_keys[best_score_index_down]
    # # TODO: minus czy plus?
    # t2_down = sorted_dic_keys[best_score_index_down-1]
    # best_threshold_down = mean([t1_down, t2_down])
    #
    # # up
    # t1_up = sorted_dic_keys[best_score_index_up]
    # # TODO: minus czy plus?
    # t2_up = sorted_dic_keys[best_score_index_up-1]
    # best_threshold_up = 0.5 + mean([t1_up, t2_up])
    #
    # return (best_threshold_down, best_threshold_up), max_score
