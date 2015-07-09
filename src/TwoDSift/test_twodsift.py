__author__ = 'agnieszka'


def build_dataset(shuffle=True, extending=True, cv=None):
    import sys
    sys.path.append('..')
    from twodsifts_dataset import TwoDSiftData
    import configuration.model as config
    data = TwoDSiftData(config.data_path, [[2], [1], [0]],
                        cv=cv, shuffle=shuffle, normal_run=extending)
    return data


def save_data(path, data_to_save):
    from pylearn2.utils.serial import save
    save(path, data_to_save)


def check_files_identity(path1, path2):
    import filecmp
    print filecmp.cmp(path1, path2)


def sample_data_to_check_preprocessing():
    from numpy import zeros
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
    import sys
    sys.path.append('..')
    import configuration.model as config
    return config.data_path


def os_test():
    import os
    print os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_sample_experiment():
    from hyperopt.pyll.stochastic import sample
    from pylearn2.config import yaml_parse
    from os.path import join
    import sys
    sys.path.append('..')
    from hyperopt_api.parser import build
    from yaml_parser import yaml_parser as yp
    from hyperopt_api.search_space import get_search_space
    import configuration.model as config
    from utils.common import get_timestamp

    # prepare all variables that don't need to be updated with each iteration
    spa = get_search_space()    # define search space over possible models

    path = config.data_path

    # obtain the yaml skelton
    with open(config.yaml_skelton_path) as f:
        default_string = f.read()

    samp = sample(spa)  # generate sample (will give a description of a model)
    mod = build(samp)   # based on description generated build an object that will fit into yaml_paser

    # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
    weight_decay_coeffs = yp.parse_weight_decay(mod)

    # generate a filename to store the best model
    pkl_filename = join(config.path_for_storing, get_timestamp + "_best.pkl")

    # create dictionary with hyper parameters
    hyper_params = {'model': yp.parse_to_yaml(mod), 'path': yp.parse_to_yaml(path),
                    'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}
    # fill the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    network = yaml_parse.load(yaml_string)

    return network

