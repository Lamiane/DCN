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

    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    # obtain the yaml skelton
    with open(config.yaml_skelton_path) as f:
        default_string = f.read()

    # define data paths
    path = config.data_path

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = yp.parse_weight_decay(mod)

    # generate a filename to store the best model
    pkl_filename = join(config.path_for_storing, current_time+"_best.pkl")

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}

    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    # misclass_error = 1
    f1_score_error = 1
    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
        print t.bold_magenta('NETWORK'), type(network)
        # train the model
        network.main_loop()

    except BaseException as e:  # TODO: this exception is to broad
        # if exception was thrown save yaml of a model that generated that exception
        with open(current_time+'.yaml', 'w') as YAML_FILE:
            YAML_FILE.write(yaml_string)
        #  write down errors description to a file
        with open(current_time+'.error', 'w') as ERROR_FILE:
            ERROR_FILE.write(traceback.format_exc())

    finally:
        if network is not None:
            try:
                # misclass_error = lowest_misclass_error(network.model)
                f1_score_error = 1 - f1_score(network)
            except BaseException as be:     # TODO: this exception is to broad
                print traceback.format_exc()
        # print t.bold_red("M_01: misclass_error for this model: "+str(misclass_error))
        # return misclass_error
        print t.bold_red("M_02: f1 score error for this model: "+str(f1_score_error))
        return f1_score_error


# based on pylearn2's scripts/plot_monitor.py
def lowest_misclass_error(model):
    this_model_channels = model.monitor.channels
    my_channel = this_model_channels['valid_softmax_misclass']   # TODO: maybe it shall be set in configuration?
    import numpy as np
    return np.min(my_channel.val_record)


def f1_score(train):
    # obtaining validating set # TODO: finally we want to have train-validation-test set. Or sth.
    valid_x = train.algorithm.monitoring_dataset['valid'].X
    valid_y = train.algorithm.monitoring_dataset['valid'].y

    import numpy as np
    from sklearn.metrics import f1_score
    import sys
    sys.path.append('..')
    from ROC.get_predictions import Predictor
    p = Predictor(train.model)

    # TODO: this for shall be INSIDE Predictor.get_prediction
    y_pred = []
    for i in len(valid_x):
        sample = np.reshape(valid_x, (1, config.data_height, config.data.width, 1))
        y_pred.append(np.argmax(p.get_prediction(sample)))

    score = f1_score(y_true=valid_y, y_pred=y_pred)

    return score


