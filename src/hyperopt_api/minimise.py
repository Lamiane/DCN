from hyperopt.pyll.stochastic import sample
from parser import build
from pylearn2.config import yaml_parse
from os.path import join
import traceback
import sys
sys.path.append('..')
from yaml_maker import yaml_parser as yp
from tmp import notify
from search_space import get_search_space
from blessings import Terminal
t = Terminal()


def objective_function(samp):
    print t.bold_cyan('SAMP'), samp

    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    # obtain the yaml skelton
    with open("example.yaml") as f:
        default_string = f.read()

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = "'h0': 0.00005,"
    if len(mod.layers) == 3:
        weight_decay_coeffs += "'h1': 0.00005,"
    weight_decay_coeffs += "\n" + "'softmax': 0.00005"

    # generate a filename to store the best model
    pkl_filename = "best.pkl"   # TODO: or maybe we want to have a best model of each model tested by hyperopt?
                                # ... TODO: we might get timestamp when entering this function and use it in filename...

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
        misclass_error = 0  # TODO: jak to wyluskac z modelu?!
    except BaseException as e:
        # if exception was thrown save yaml of a model that generated that exception
        with open('0000.yaml', 'w') as YAML_FILE:   # TODO: this filename is just wrong! Use timestamp
            YAML_FILE.write(yaml_string)
        #  write down errors description to a file
        with open('0000_error', 'w') as ERROR_FILE:     # TODO: this filename is just wrong! Use timestamp
            ERROR_FILE.write(traceback.format_exc())
    finally:
        return misclass_error