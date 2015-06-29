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
spa = get_search_space()    # define search space over possible models

# define data paths
dirpath = '../../data'
path = [join(dirpath, '2RH1_actives_2dfp.dat'),
        join(dirpath, '2RH1_inactives_2dfp.dat')]

# path = [join(dirpath, '2RH1_actives_2dfp.dat'),
#         join(dirpath, '2RH1_inactives_2dfp.dat'),
#         join(dirpath, '2RH1_middle_2dfp.dat')]

# obtain the yaml skelton
with open("example.yaml") as f:
    default_string = f.read()

# for each sample that will be generated from search space space
for i in xrange(20):
    print t.bold_red('ITERATION:'), t.bold_red(str(i))

    samp = sample(spa)  # generate sample (will give a description of a model)
    print t.bold_cyan('SAMP'), samp

    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser
    print t.bold_blue('MODEL'), mod

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path)}

    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    # saving the yaml for later analysis
    with open('generated_'+str(i)+'.yaml', 'w') as g:
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

# play a melody so everyone knows the experiment has finished
notify()
