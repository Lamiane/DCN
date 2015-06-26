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

# prepare all variables that don't need to be updated with each iteration

spa = get_search_space()    # define search space

# creating a dictionary with hyperparameters
dirpath = '../../data'
path = [join(dirpath, '2RH1_actives_2dfp.dat'),
        join(dirpath, '2RH1_inactives_2dfp.dat'),
        join(dirpath, '2RH1_middle_2dfp.dat')]

# obtaining the yaml skelton
with open(sys.argv[1]) as f:
    yaml_string = f.read()

try:
    # creating the model based on a yaml
    network = yaml_parse.load(yaml_string)

    # training the model
    print t.bold_magenta('NETWORK'), type(network)
    network.main_loop()
except BaseException as e:
    with open('A01.yaml', 'w') as YAML_FILE:
        YAML_FILE.write(yaml_string)
    with open('A01_error', 'w') as ERROR_FILE:
        ERROR_FILE.write(traceback.format_exc())


notify()
