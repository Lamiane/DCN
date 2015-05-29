from twodsifts_dataset import *
from os.path import join


def build_dataset():
    dirpath = '../../data'
    data = TwoDSiftData([join(dirpath, '2RH1_actives_2dfp.dat'),
                         join(dirpath, '2RH1_inactives_2dfp.dat'),
                         join(dirpath, '2RH1_middle_2dfp.dat')], [[2], [1], [0]])
    return data