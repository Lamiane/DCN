__author__ = 'nex'


def f1_score_1threshold_get_value(train):
    from numpy import argmax
    import sys
    sys.path.append('..')
    from algorithm_extensions.symmetric_threshold import SymmetricThresholdWRTF1Score

    try:
        # finding F1Score1Threshold extension in train.extensions
        f1_score_ext = None
        if 'extensions' in dir(train):
            for element in train.extensions:
                if isinstance(element, SymmetricThresholdWRTF1Score):
                    f1_score_ext = element
                    break
        elif 'cv_extensions' in dir(train):
            for element in train.cv_extensions:
                if isinstance(element, SymmetricThresholdWRTF1Score):
                    f1_score_ext = element
                    break

        best_f1_score = max(f1_score_ext.score_list)
        threshold = f1_score_ext.threshold_list[argmax(f1_score_ext.score_list)]
        return best_f1_score, threshold

    except AttributeError as ae:
        # return if F1Score extension hasn't been found
        print "This pylearn.train.Train object doesn't use " \
              "algorithm_extensions.symmetric_threshold.SymmetricThresholdWRTF1Score " \
              "F1 score hasn't been calculated. Please include SymmetricThresholdWRTF1Score extension in yaml " \
              "if you need F1 score calculated"
        raise ae


def f1_score_get_value(train):
    import sys
    sys.path.append('..')
    from algorithm_extensions.no_threshold import F1Score

    try:
        # finding F1Score extension in train.extensions
        f1_score_ext = None
        if 'extensions' in dir(train):
            for element in train.extensions:
                if isinstance(element, F1Score):
                    f1_score_ext = element
                    break
        elif 'cv_extensions' in dir(train):
            for element in train.cv_extensions:
                if isinstance(element, F1Score):
                    f1_score_ext = element
                    break

        best_f1_score = max(f1_score_ext.score_list)

        return best_f1_score
    except AttributeError as ae:
        # return if F1Score extension hasn't been found
        print "This pylearn.train.Train object doesn't use algorithm_extensions.f1_score.F1Score extension. " \
              "F1 score hasn't been calculated. Please provide include F1Score extension in yaml " \
              "if you need F1 score calculated"
        raise ae


def lowest_misclass_error_get_value(model):
    this_model_channels = model.monitor.channels
    my_channel = this_model_channels['valid_softmax_misclass']  # TODO: maybe it shall be set in configuration?
    import numpy as np

    return np.min(my_channel.val_record)