__author__ = 'agnieszka'

from twodsifts_dataset import *
from os.path import join
from pylearn2.utils.serial import save


def build_dataset():
    dirpath = '../../data'
    data = TwoDSiftData([join(dirpath, '2RH1_actives_2dfp.dat'),
                         join(dirpath, '2RH1_inactives_2dfp.dat'),
                         join(dirpath, '2RH1_middle_2dfp.dat')], [[2], [1], [0]], cv=[5, [0, 1, 2, 3]])
    return data


def save_data(path, data_to_save):
    save(path, data_to_save)


def check_files_identity(path1, path2):
    import filecmp
    filecmp.cmp(path1, path2)