__author__ = 'nex'
from extract_all_numbers import retrieve_lines_with_information

filename = 'MEAN'
retrieve_lines_with_information(filename)

from extract_all_numbers import run_global_zusammen
file_list = ['min_cut', 'soft_cut', 'mean_cut']
legend_list = ['minimum', 'softmax', 'mean']
run_global_zusammen(file_list, legend_list, 'GLOBAL')

from extract_all_numbers import turn_into_structure, check_sanity_of_dic, analyse_1
filename = 'mean_cut'
d = turn_into_structure(filename)
check_sanity_of_dic(d)
analyse_1(d)

from extract_all_numbers import turn_into_structure, analyse_2
filename = 'mean_cut'
d = turn_into_structure(filename)
analyse_2(d)

from extract_all_numbers import variance_zusammen
file_list = ['min_cut', 'soft_cut', 'mean_cut']
variance_zusammen(file_list, 'VARIANCE')

from extract_all_numbers import learning_1_best
filename = 'mean_cut'
output_name = 'MEAN_LEARNING'
title = 'MEAN'
learning_1_best(filename, output_name, title)

from extract_all_numbers import max_in_epoch
filename = 'soft_cut'
title = 'SOFTMAX'
output_name = 'SOFT_MAX_IN_EPOCH'
max_in_epoch(filename, output_name, title)

from extract_all_numbers import number_of_epochs
filename = 'soft_cut'
title = 'SOFTMAX'
output_name = 'SOFT_N_EPOCHS'
number_of_epochs(filename, output_name, title)

from extract_all_numbers import score_layers
file_list = ['min_cut', 'soft_cut', 'mean_cut']
output_name = 'SCORE_LAYER'
legend_list = ['MINIMUM', 'SOFTMAX', 'MEAN']
score_layers(file_list, output_name, legend_list)

from extract_all_numbers import prec_rec_score_combination_function
file_list = ['min_cut', 'soft_cut', 'mean_cut']
output_name = 'SCORE_BY_COMB'
legend_list = ['MINIMUM', 'SOFTMAX', 'MEAN']
prec_rec_score_combination_function(file_list, output_name, legend_list)

from extract_all_numbers import mcc_statistics, turn_into_structure
filename = 'MININUM'
info = turn_into_structure(filename)
mcc_statistics(info)

from extract_all_numbers import mcc_compare_all_three, turn_into_structure
minimum = turn_into_structure('min_cut')
softmax = turn_into_structure('soft_cut')
mean = turn_into_structure('mean_cut')
output_name = 'MCC_SCORE_BY_COMB'
legend_list = ['MINIMUM', 'SOFTMAX', 'MEAN']
mcc_compare_all_three([minimum, softmax, mean], output_name, legend_list)

from extract_all_numbers import mcc_score_layers
file_list = ['min_cut', 'soft_cut', 'mean_cut']
output_name = 'MCC_SCORE_LAYER'
legend_list = ['MINIMUM', 'SOFTMAX', 'MEAN']
mcc_score_layers(file_list, output_name, legend_list)
