__author__ = 'agnieszka'
from pylearn2.train_extensions import TrainExtension
from sklearn.metrics import f1_score
import sys
sys.path.append('..')
from get_predictions import Predictor


class F1Score(TrainExtension):
    def __init__(self, save_best_model_path=None):
        self.predictor = None   # TODO: perhaps predictor shouldn't be a class field
                                # ...TODO: and should be initialised every time on_monitoring is called?
        self.score_list = []
        self.debug = True
        self.saving_path=save_best_model_path

    def setup(self, model, dataset, algorithm):
        self.predictor = Predictor(model)

    def on_monitor(self, model, dataset, algorithm):
        import numpy as np
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

        if self.saving_path is not None:
            if max(self.score_list) == score:
                pass
                # TODO: saving here

        print "F1 score:", score

        ## <debug> POCHA
        if self.debug:
            import sys
            sys.path.append('..')
            from utils.common import get_timestamp
            current_time = get_timestamp()
            np.array(valid_x).dump(str(current_time)+"_valid_x")
            np.array(valid_y).dump(str(current_time)+"_valid_y")
            np.array(y_classes).dump(str(current_time)+"_y_classes")

            self.debug = False

        ## </debug> POCHA

    # we don't need it, let's inherit the default empty method
    # def on_save(self, model, dataset, algorithm):
    #     pass

