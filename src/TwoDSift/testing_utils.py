__author__ = 'agnieszka'

from twodsifts_dataset import *
from os.path import join
from pylearn2.utils.serial import save
from numpy import zeros


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