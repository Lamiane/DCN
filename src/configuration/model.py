__author__ = 'agnieszka'
from os.path import join


def get_example_yaml_path():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(path, "example.yaml")

# # # STATIC VALUES # # #
#########################

# TODO: obudowac geterami? zrobic z tego klase? zostawic jak jest?

data_height = 18
data_width = 3492   # 3474 + 2*9

data_dir_path = '../../data'
data_path = [join(data_dir_path, '2RH1_actives_2dfp.dat'),
             join(data_dir_path, '2RH1_inactives_2dfp.dat')]

# data_path = [join(data_dir_path, '2RH1_actives_2dfp.dat'),
#              join(data_dir_path, '2RH1_inactives_2dfp.dat'),
#              join(data_dir_path, '2RH1_middle_2dfp.dat')]

yaml_skelton_path = get_example_yaml_path()
path_for_storing = '../../data/generated'
