__author__ = 'agnieszka'
from hyperopt.pyll.stochastic import sample
from parser import build
from pylearn2.config import yaml_parse
from os.path import join
import traceback
import sys
sys.path.append('..')
from yaml_parser import yaml_parser as yp
from utils.common import notify, get_timestamp
from hyperopt_api.search_space import get_search_space
from blessings import Terminal
t = Terminal()
import configuration.model as config

# prepare all variables that don't need to be updated with each iteration
spa = get_search_space()    # define search space over possible models

# define data paths
path = config.data_path

# obtain the yaml skelton
with open(config.yaml_skelton_path) as f:
    default_string = f.read()

# for each sample that will be generated from search space space
for i in xrange(20):
    timestamp = get_timestamp()

    print t.bold_red('ITERATION:'), t.bold_red(str(i)), "started at: ", timestamp

    samp = sample(spa)  # generate sample (will give a description of a model)
    print t.bold_cyan('SAMP'), samp

    mod = build(samp)   # based on description generated build an object that will fit into yaml_parser
    print t.bold_blue('MODEL'), mod

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = yp.parse_weight_decay(mod)

    # generate a filename to store the best model
    pkl_filename = join(config.path_for_storing, timestamp+"best_"+str(i)+'_'+".pkl")

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}

    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    # saving the yaml for later analysis
    yaml_path = join(config.path_for_storing, timestamp+'generated_'+str(i)+'.yaml')
    with open(yaml_path, 'w') as g:
        g.write(yaml_string)

    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
        print t.bold_magenta('NETWORK'), type(network)

        # train the model
        network.main_loop()
    except BaseException as e:
        # if exception was thrown save yaml of a model that generated that exception
        with open('0000'+str(i)+'.yaml', 'w') as YAML_FILE:
            YAML_FILE.write(yaml_string)
        #  write down errors description to a file
        with open('0000'+str(i)+'_error', 'w') as ERROR_FILE:
            ERROR_FILE.write(traceback.format_exc())

# play a melody so everyone knows the testing has finished
notify()
