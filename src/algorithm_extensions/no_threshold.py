__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
from pylearn2.train import SerializationGuard
from pylearn2.utils import serial
from sklearn.metrics import f1_score
import sys
sys.path.append('..')
from get_predictions import Predictor


class F1Score(TrainExtension):
    def __init__(self, save_best_model_path=None, save=False):
        self.predictor = None
        self.score_list = []
        self.debug = True
        self.saving_path = save_best_model_path
        self.save = save

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm):
        import numpy as np
        # this shall never happen but better safe than sorry
        if self.predictor is None:
            self.setup(model, dataset, algorithm)

        # obtaining validating set #
        valid_x = algorithm.monitoring_dataset['valid'].X
        valid_y = algorithm.monitoring_dataset['valid'].y
        y_pred = self.predictor.get_predictions(valid_x)
        y_classes = [np.argmax(pred) for pred in y_pred]
        score = f1_score(y_true=valid_y, y_pred=y_classes)
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

        print "F1 score:", score
