__author__ = 'agnieszka'
from os.path import join, dirname, realpath, abspath


def get_example_yaml_path():
    path_d = dirname(realpath(__file__))
    return join(path_d, "example.yaml")


def get_data_scheme_path():
    path_d = dirname(realpath(__file__))
    return join(path_d, 'data_scheme.yaml')


def get_data_path():
    path_d = dirname(realpath(__file__))
    return abspath(join(path_d, "../../data"))

# # # STATIC VALUES # # #
#########################

# active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
# nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
# middle    -1    [[ 0. 0. 1. ]]

# TODO: obudowac geterami? zrobic z tego klase? zostawic jak jest?

data_height = 18
data_width = 3492   # 3474 + 2*9

data_dir_path = get_data_path()

# data_path = [join(data_dir_path, '2RH1_actives_2dfp.dat'),
#              join(data_dir_path, '2RH1_inactives_2dfp.dat')]
#
# data_path = [join(data_dir_path, '2RH1_actives_2dfp.dat'),
#              join(data_dir_path, '2RH1_inactives_2dfp.dat'),
#              join(data_dir_path, '2RH1_middle_2dfp.dat')]

# DEFAULT
data_dict = {'labeled_paths': [join(data_dir_path, '2RH1_actives_2dfp.dat'),
                               join(data_dir_path, '2RH1_inactives_2dfp.dat')],
             'labeled_values': [[1], [0]],
             'middle_paths': [join(data_dir_path, '2RH1_middle_2dfp.dat')],
             'middle_values': [[-1]]
             }

# NAIVE APPROACH
# data_dict = {'labeled_paths': [join(data_dir_path, '2RH1_actives_2dfp.dat'),
#                                join(data_dir_path, '2RH1_actives_2dfp.dat'),
#                                join(data_dir_path, '2RH1_inactives_2dfp.dat'),
#                                join(data_dir_path, '2RH1_inactives_2dfp.dat'),
#                                join(data_dir_path, '2RH1_middle_2dfp.dat'),
#                                join(data_dir_path, '2RH1_middle_2dfp.dat')
#                                ],
#              'labeled_values': [[1], [1], [0], [0], [1], [0]],
#              'middle_paths': [],
#              'middle_values': []
#              }


seed = 1337

yaml_skelton_path = get_example_yaml_path()
data_yaml_scheme = get_data_scheme_path()
path_for_storing = join(get_data_path(), "generated")

###########################################
# # # C R O S S - V A L I D A T I O N # # #
###########################################
number_of_cross_validation_parts_outer = 5
number_of_cross_validation_parts_inner = 5
actives_path = '/home/nex/Dropbox (GMUM)/ujDOK1/2DSIFt/2DSIFtCNN/2d_sift/data/beta2_acives_full_pickled_2dfp.npy'
nonactives_path = '/home/nex/Dropbox (GMUM)/ujDOK1/2DSIFt/2DSIFtCNN/2d_sift/data/beta2_inactive_full_pickle_2dfp.npy'
