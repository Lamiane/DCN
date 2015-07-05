from hyperopt.pyll.stochastic import sample
from parser import build
from pylearn2.config import yaml_parse
from os.path import join
import traceback
import sys
sys.path.append('..')
from yaml_maker import yaml_parser as yp
from blessings import Terminal
t = Terminal()
from utilss.common import get_timestamp


def run(max_evals=10):
    from hyperopt import fmin, tpe
    from hyperopt_api.search_space import get_search_space
    best = fmin(objective_function, get_search_space(), algo=tpe.suggest, max_evals=max_evals)
    return best


def objective_function(samp):
    current_time = get_timestamp()

    print t.bold_cyan('SAMP'), samp

    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    # obtain the yaml skelton
    with open("example.yaml") as f:
        default_string = f.read()

    # define data paths
    dirpath = '../../data'
    path = [join(dirpath, '2RH1_actives_2dfp.dat'),
            join(dirpath, '2RH1_inactives_2dfp.dat')]

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = "'h0': 0.00005,"
    if len(mod.layers) == 3:
        weight_decay_coeffs += "'h1': 0.00005,"
    weight_decay_coeffs += "\n" + "'softmax': 0.00005"

    # generate a filename to store the best model
    pkl_filename = current_time+"_best.pkl"

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}

    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    misclass_error = 1
    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
        print t.bold_magenta('NETWORK'), type(network)

        # train the model
        network.main_loop()
        misclass_error = lowest_misclass_error(network)
    except BaseException as e:
        # if exception was thrown save yaml of a model that generated that exception
        with open(current_time+'.yaml', 'w') as YAML_FILE:
            YAML_FILE.write(yaml_string)
        #  write down errors description to a file
        with open(current_time+'.error', 'w') as ERROR_FILE:
            ERROR_FILE.write(traceback.format_exc())
    finally:
        print t.bold_red("misclass_error for this model #01: "+str(misclass_error))
        return misclass_error


# based on pylearn2's scripts/plot_monitor.py
def lowest_misclass_error(model):
    this_model_channels = model.monitor.channels
    my_channel = this_model_channels['valid_softmax_misclass'] # TODO: wywalic do konfiguracji
    return my_channel.val_record[-1]    # AFAIK, last ist best.


def pkl2model(filepath):
    from pylearn2.utils import serial
    model = serial.load(filepath)
    return model
