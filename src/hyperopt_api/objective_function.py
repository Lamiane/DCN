__author__ = 'agnieszka'
from parser import build
from pylearn2.config import yaml_parse
from os.path import join
import traceback
import sys

sys.path.append('..')
from yaml_parser import yaml_parser as yp
from blessings import Terminal

t = Terminal()
from utils.common import get_timestamp
import configuration.model as config


def objective_function(samp):
    current_time = get_timestamp()

    print t.bold_cyan('SAMP'), samp

    mod = build(samp)  # based on description generated build an object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    # obtain the yaml skelton
    with open(config.yaml_skelton_path) as f:
        default_string = f.read()

    # define data paths
    path = config.data_path

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = yp.parse_weight_decay(mod)

    # generate a filename to store the best model
    pkl_filename = join(config.path_for_storing, current_time + "_best.pkl")

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}

    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    network = None
    # misclass_error = 1
    f1_score_error = 1
    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
        print t.bold_magenta('NETWORK'), type(network)
        # train the model
        network.main_loop()

    except BaseException:  # TODO: this exception is to broad
        # if exception was thrown save yaml of a model that generated that exception
        with open(current_time + '.yaml', 'w') as YAML_FILE:
            YAML_FILE.write(yaml_string)
        # write down errors description to a file
        with open(current_time + '.error', 'w') as ERROR_FILE:
            ERROR_FILE.write(traceback.format_exc())

    finally:
        if network is not None:
            try:
                # misclass_error = lowest_misclass_error(network.model)
                # f1_score_error = 1 - f1_score(network)
                f1_score_error, threshold = f1_score_1threshold(network)
                f1_score_error = 1 - f1_score_error
            except BaseException:  # TODO: this exception is to broad
                with open(current_time + '_f1_error', 'w') as ERROR_FILE:
                    ERROR_FILE.write(traceback.format_exc())

        # print t.bold_red("M_01: misclass_error for this model: "+str(misclass_error))
        # return misclass_error
        print t.bold_red("M_02: f1 score error for this model: " + str(f1_score_error))
        return f1_score_error


# based on pylearn2's scripts/plot_monitor.py
def lowest_misclass_error(model):
    this_model_channels = model.monitor.channels
    my_channel = this_model_channels['valid_softmax_misclass']  # TODO: maybe it shall be set in configuration?
    import numpy as np

    return np.min(my_channel.val_record)


def f1_score(train):
    import sys
    sys.path.append('..')
    from algorithm_extensions.f1_score import F1Score

    try:
        # finding F1Score extension in train.extensions
        f1_score_ext = None
        for element in train.extensions:
            if isinstance(element, F1Score):
                f1_score_ext = element
                break

        best_f1_score = max(f1_score_ext.score_list)

        return best_f1_score
    except AttributeError as ae:
        # return if F1Score extension hasn't been found
        print "This pylearn.train.Train object doesn't use extensions.f1_score.F1Score extension. " \
              "F1 score hasn't been calculated. Please provide include F1Score extension in yaml " \
              "if you need F1 score calculated"
        raise ae


def f1_score_1threshold(train):
    from numpy import argmax
    import sys
    sys.path.append('..')
    from algorithm_extensions.f1_score import F1Score1Threshold

    try:
        # finding F1Score extension in train.extensions
        f1_score_ext = None
        for element in train.extensions:
            if isinstance(element, F1Score1Threshold):
                f1_score_ext = element
                break

        best_f1_score = max(f1_score_ext.score_list)
        threshold = f1_score_ext.threshold_list[argmax(f1_score_ext.score_list)]

        print t.bold_red("D_OF1: Best score for this model: "+str(best_f1_score))
        print t.bold_red("D_OF1: Obtained for threshold: "+str(threshold))

        return best_f1_score, threshold
    except AttributeError as ae:
        # return if F1Score extension hasn't been found
        print "This pylearn.train.Train object doesn't use extensions.f1_score.F1Score extension. " \
              "F1 score hasn't been calculated. Please provide include F1Score extension in yaml " \
              "if you need F1 score calculated"
        raise ae

