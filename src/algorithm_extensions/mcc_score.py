__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial
import sys
sys.path.append('..')
from get_predictions import Predictor
from utils.casting import types_dict, pred_and_trues_to_type_dict
from utils import values


# valid_y - the true classes, predictions - well, predicted values (not classes!)
def mcc_score(true_y, predictions):
    stat_dic = {}
    import numpy as np
    print true_y.shape
    print predictions.shape

    if np.all(el == 0 or el == 1 for el in np.concatenate(true_y, predictions)):
        stat_dic = pred_and_trues_to_type_dict(true_y, predictions)
    else:
        stat_dic = types_dict(true_y, predictions, threshold=0.5)
    tp = stat_dic[values.TP]
    tn = stat_dic[values.TN]
    fp = stat_dic[values.FP]
    fn = stat_dic[values.FN]
    numerator = (tp * tn) + (fp * fn)
    denominator_2 = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denominator = 1  # this should stay if any of above sums is equal to 0
    if denominator_2 != 0:
        denominator = np.sqrt(denominator_2)
    mcc = numerator/denominator
    return mcc


class MCC(TrainExtension):
    def __init__(self, save_best_model_path=None, save=False):
        self.predictor = None
        self.score_list = []
        self.debug = True
        self.saving_path = save_best_model_path
        self.save = save

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm):
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        # obtaining validating set #
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y
        y_pred = self.predictor.get_predictions(valid_x)
        score = mcc_score(true_y=valid_y, predictions=y_pred)
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

        print "\nMCC score (no threshold):", score
