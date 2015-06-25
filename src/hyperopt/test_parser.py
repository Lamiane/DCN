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
with open("example.yaml") as f:
    default_string = f.read()

for i in xrange(1):
    print t.bold_red('ITERATION:'), t.bold_red(str(i))

    samp = sample(spa)  # generate sample
    print t.bold_cyan('SAMP'), samp

    mod = build(samp)   # build object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path)}

    # filling the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    # saving the yaml for later analysis
    with open('generated_'+str(i)+'.yaml', 'w') as g:
        g.write(yaml_string)
    try:
        # creating the model based on a yaml
        network = yaml_parse.load(yaml_string)

        # training the model
        print t.bold_magenta('NETWORK'), type(network)
        network.main_loop()
    except BaseException as e:
        with open('0000'+str(i)+'.yaml', 'w') as YAML_FILE:
            YAML_FILE.write(yaml_string)
        with open('0000'+str(i)+'_error', 'w') as ERROR_FILE:
            ERROR_FILE.write(traceback.format_exc())


notify()
