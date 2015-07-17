__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
import numpy as np
from sklearn.metrics import f1_score
import sys
sys.path.append('..')
from get_predictions import Predictor


class F1Score(TrainExtension):
    def __init__(self):
        self.predictor = None   # TODO: perhaps predictor shouldn't be a class field
                                # ...TODO: and should be initialised every time on_monitoring is called?
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
        print "F1 score:", score

    # we don't need it, let's inherit the default empty method
    # def on_save(self, model, dataset, algorithm):
    #     pass


class Precision(TrainExtension):
    def __init__(self):
        self.predictor = None
        self.precision_list = []

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm, stat_dic=None):
        import sys
        sys.path.append('..')
        from utils import values

        if stat_dic is None:
            from utils.casting import label_lists2types
            # obtaining validating set
            valid_x = algorithm.monitoring_dataset['valid'].X
            valid_y = algorithm.monitoring_dataset['valid'].y
            y_pred = self.predictor.get_predictions(valid_x)
            stat_dic = label_lists2types(valid_y, y_pred)

        # precision = TP/(TP + FP)
        precision = stat_dic[values.TP]/(stat_dic[values.TP] + stat_dic[values.FP])

        self.precision_list.append(precision)
        print "precision:", precision


class Recall(TrainExtension):
    def __init__(self):
        self.predictor = None
        self.recall_list = []

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm, stat_dic=None):
        import sys
        sys.path.append('..')
        from utils import values

        if stat_dic is None:
            from utils.casting import label_lists2types
            # obtaining validating set
            valid_x = algorithm.monitoring_dataset['valid'].X
            valid_y = algorithm.monitoring_dataset['valid'].y
            y_pred = self.predictor.get_predictions(valid_x)
            stat_dic = label_lists2types(valid_y, y_pred)

        # recall = TP/(TP + FN)
        recall = stat_dic[values.TP]/(stat_dic[values.TP] + stat_dic[values.FN])

        self.recall_list.append(recall)
        print "recall:", recall


class Accuracy(TrainExtension):
    def __init__(self):
        self.predictor = None
        self.accuracy_list = []

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm, stat_dic=None):
        import sys
        sys.path.append('..')
        from utils import values

        if stat_dic is None:
            from utils.casting import label_lists2types
            # obtaining validating set
            valid_x = algorithm.monitoring_dataset['valid'].X
            valid_y = algorithm.monitoring_dataset['valid'].y

            y_pred = self.predictor.get_predictions(valid_x)
            stat_dic = label_lists2types(valid_y, y_pred)

        # accuracy = (TP + TN) / TOTAL
        accuracy = (stat_dic[values.TP]+stat_dic[values.TN])/sum(stat_dic.values())

        self.accuracy_list.append(accuracy)
        print "accuracy:", accuracy