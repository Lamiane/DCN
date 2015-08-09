__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
from get_predictions import Predictor
import sys
sys.path.append('..')
from symmetric_threshold import SymmetricThresholdWRTF1Score


class StatisticsNoThreshold(TrainExtension):
    def __init__(self, call_list):
        """
        :type call_list: list containing objects to calculate certain statistics
        """
        self.predictor = None
        self.call_list = call_list

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)
        for extension in self.call_list:
            extension.setup(model, dataset, algorithm)

    def on_monitor(self, model, dataset, algorithm):
        import sys
        sys.path.append('..')
        from utils.casting import label_lists2types
        from utils import values

        print '\nSHOWING STATISTICS FOR NO THRESHOLD'

        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y
        y_pred = self.predictor.get_predictions(valid_x)
        stat_dic = label_lists2types(valid_y, y_pred)

        print values.TP, ':', stat_dic[values.TP], '\t\t', values.TN, ':', stat_dic[values.TN], '\n', \
            values.FP, ':', stat_dic[values.FP], '\t\t', values.FN, ':', stat_dic[values.FN], '\n', \
            values.FNP, ':', stat_dic[values.FNP], '\t\t', values.FNN, ':', stat_dic[values.FNN]

        for extension in self.call_list:
            # TODO: consider using inspect.getargspec()
            try:
                extension.on_monitor(model, dataset, algorithm, stat_dic)
            except TypeError:
                extension.on_monitor(model, dataset, algorithm)


class StatisticsSymmetricThreshold(TrainExtension):
    def __init__(self, call_list):
        """
        :type call_list: list containing objects to calculate certain statistics
        """
        self.predictor = None
        self.symmetric_threshold = None
        self.call_list = call_list

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)
        self.symmetric_threshold = SymmetricThresholdWRTF1Score()
        for extension in self.call_list:
            extension.setup(model, dataset, algorithm)

    def on_monitor(self, model, dataset, algorithm):
        import sys
        sys.path.append('..')
        from utils.casting import label_lists2types
        from utils import values

        print '\nSHOWING STATISTICS FOR SYMMETRIC THRESHOLD'

        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y
        y_pred = self.predictor.get_predictions(valid_x)
        threshold, score = self.symmetric_threshold.compute_optimal_threshold_and_score(valid_y, y_pred)
        print "Best threshold", threshold, '\ncorresponding F1Score:', score
        stat_dic = label_lists2types(valid_y, y_pred, sym_t=threshold)

        print values.TP, ':', stat_dic[values.TP], '\t\t', values.TN, ':', stat_dic[values.TN], '\n', \
            values.FP, ':', stat_dic[values.FP], '\t\t', values.FN, ':', stat_dic[values.FN], '\n', \
            values.FNP, ':', stat_dic[values.FNP], '\t\t', values.FNN, ':', stat_dic[values.FNN]

        for extension in self.call_list:
            # TODO: consider using inspect.getargspec()
            try:
                extension.on_monitor(model, dataset, algorithm, stat_dic)
            except TypeError:
                extension.on_monitor(model, dataset, algorithm)

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
        if stat_dic[values.TP] == 0:
            precision = 0
        else:
            precision = float(stat_dic[values.TP])/(stat_dic[values.TP] + stat_dic[values.FP])

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
        if stat_dic[values.TP] == 0:
            recall = 0
        else:
            recall = float(stat_dic[values.TP])/(stat_dic[values.TP] + stat_dic[values.FN])

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
        if (stat_dic[values.TP] + stat_dic[values.TN]) == 0:
            accuracy = 0
        else:
            accuracy = float(stat_dic[values.TP]+stat_dic[values.TN])/sum(stat_dic.values())

        self.accuracy_list.append(accuracy)
        print "accuracy:", accuracy