__author__ = 'agnieszka'

from twodsifts_dataset import *
from os.path import join
from pylearn2.utils.serial import save
from numpy import zeros


def build_dataset(shuffle=True, extending=True, cv=[5, [0, 1, 2, 3]]):
    dirpath = '../../data'
    data = TwoDSiftData([join(dirpath, '2RH1_actives_2dfp.dat'),
                         join(dirpath, '2RH1_inactives_2dfp.dat'),
                         join(dirpath, '2RH1_middle_2dfp.dat')], [[2], [1], [0]],
                        cv=cv, shuffle=shuffle, normal_run=extending)
    return data


def save_data(path, data_to_save):
    save(path, data_to_save)


def check_files_identity(path1, path2):
    import filecmp
    print filecmp.cmp(path1, path2)


def sample_data_to_check_preprocessing():
    shape = (1, 6, 9, 1)
    sample_data = zeros(shape)
    s00 = [i for i in xrange(9)]
    s01 = [i+10 for i in s00]
    s10 = [i+50 for i in s00]
    s11 = [i+10 for i in s10]
    s20 = [i+100 for i in s00]
    s21 = [number+10 for number in s20]

    sample_data[0, 0, :, 0] = s00
    sample_data[0, 1, :, 0] = s01
    sample_data[0, 2, :, 0] = s10
    sample_data[0, 3, :, 0] = s11
    sample_data[0, 4, :, 0] = s20
    sample_data[0, 5, :, 0] = s21

    return sample_data


def file_list():
    dirpath = '../../data'
    return [join(dirpath, '2RH1_actives_2dfp.dat'),
            join(dirpath, '2RH1_inactives_2dfp.dat'),
            join(dirpath, '2RH1_middle_2dfp.dat')]


def os_test():
    print os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_sample_experiment():
    from hyperopt.pyll.stochastic import sample
    from parser import build
    from pylearn2.config import yaml_parse
    from os.path import join
    import sys
    sys.path.append('..')
    from yaml_maker import yaml_parser as yp
    from search_space import get_search_space

    # prepare all variables that don't need to be updated with each iteration
    spa = get_search_space()    # define search space over possible models

    # define data paths
    dirpath_data = '../../data'

    path = [join(dirpath_data, '2RH1_actives_2dfp.dat'),
            join(dirpath_data, '2RH1_inactives_2dfp.dat'),
            join(dirpath_data, '2RH1_middle_2dfp.dat')]

    # obtain the yaml skelton
    dirpath_yaml = '../hyperopt'
    with open(join(dirpath_yaml, "example.yaml")) as f:
        default_string = f.read()

    samp = sample(spa)  # generate sample (will give a description of a model)
    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = "'h0': 0.00005,"
    if len(mod.layers) == 3:
        weight_decay_coeffs += "'h1': 0.00005,"
    weight_decay_coeffs += "\n" + "'softmax': 0.00005"
    # generate a filename to store the best model
    pkl_filename = "best_.pkl"

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}
    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    network = yaml_parse.load(yaml_string)

    return network

