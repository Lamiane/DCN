__author__ = 'nex'
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial
import sys
sys.path.append('..')
from no_threshold import F1Score
from get_predictions import Predictor
from utils.casting import types_dict
from utils import values

class ROC_Yoduen(F1Score):
    def __init__(self, save_best_model_path=None, save=False):
        super(ROC_Yoduen, self).__init__()
        self.threshold_list = []
        self.saving_path = save_best_model_path
        self.save = save

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm):
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        print 'into the roc!'

        # obtaining validating set # TODO: finally we want to have train-validation-test set. Or sth.
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y
        predictions = self.predictor.get_predictions(valid_x)
        best_threshold, best_score = self.compute_optimal_threshold_and_score(valid_y, predictions)

        self.threshold_list.append(best_threshold)
        self.score_list.append(best_score)

        if self.saving_path is not None and self.save:
            if max(self.score_list) == best_score:
                try:
                    # Make sure that saving does not serialize the dataset
                    dataset._serialization_guard = SerializationGuard()
                    # TODO: this should actually be: save_path = self.saving_path
                    save_path = 'best_model_roc_youden.model'
                    serial.save(save_path, model,
                                on_overwrite='backup')
                finally:
                    dataset._serialization_guard = None

        stat_dic = types_dict(valid_y, predictions, threshold=best_threshold)

        print '\nSHOWING STATISTICS FOR ROC with Youden metric'
        print "ROC using Youden metric\nscore:", best_score, "\ncorresponding threshold:", best_threshold
        print values.TP, ':', stat_dic[values.TP], '\t\t', values.TN, ':', stat_dic[values.TN], '\n', \
            values.FP, ':', stat_dic[values.FP], '\t\t', values.FN, ':', stat_dic[values.FN], '\n', \
            values.FNP, ':', stat_dic[values.FNP], '\t\t', values.FNN, ':', stat_dic[values.FNN]

        precision = float(stat_dic[values.TP])/(stat_dic[values.TP] + stat_dic[values.FP])
        recall = float(stat_dic[values.TP])/(stat_dic[values.TP] + stat_dic[values.FN])
        f1score = 0
        if precision+recall != 0:
            f1score = 2*precision*recall/(precision+recall)
        print 'precision:',precision
        print "recall:", recall
        print "f1score", f1score


    @staticmethod
    def compute_optimal_threshold_and_score(true_y, predictions):
        axis = sorted([[pred[0][1], label] for pred, label in zip(predictions, true_y)])

        # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
        # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
        # middle    -1    [[ 0. 0. 1. ]]

        print 'into the insides of roc!'

        actives = sum([1 for arr_pred_lab in axis if arr_pred_lab[1] == 1])
        nonactives = len(axis) - actives

        # threshold is zero, it means we label ALL the samples as negatives
        TP = 0
        FP = 0
        TN = nonactives
        FN = actives

        best_score = 0
        best_threshold = float(0)
        next_pred_after_best_threshold = 0
        update_next = False
        print 'axis length', len(axis)
        for prediction, label in axis:
            if label[0] == 1:
                TP -= 1     # after moving threshold we have one well classified positive example less
                FN += 1     # and that means we classify wrongly one positive example more
            else:   # label is 0 - means nonactive
                TN += 1     # we have one well classified negative example more
                FP -= 1     # so that's one wrongly classified negative example less
            # calculating score according to Youden's metric
            score = (float(TP)/(float(TP) + float(FN))) - (float(FP)/(float(FP) + float(TN)))
            print 'TP:', TP, '\tFP:', FP, '\nFN:', FN, '\tTN:', TN
            print 'score:', score, 'threshold:', prediction, '\n\n'

            if update_next:
                update_next = False
                next_pred_after_best_threshold = prediction
            if score == best_score:
                pass
                #print 'TP:', TP, '\tFP:', FP, '\nFN:', FN, '\tTN:', TN
            if score > best_score:
                #print 'TP:', TP, '\tFP:', FP, '\nFN:', FN, '\tTN:', TN
                best_score = score
                best_threshold = prediction
                update_next = True

        # compute optimal threshold
        best_threshold = (best_threshold + next_pred_after_best_threshold)/2.0
        return best_threshold, best_score
