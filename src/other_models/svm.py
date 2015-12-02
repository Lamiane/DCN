from sklearn import svm
from pylearn2.config import yaml_parse
from numpy import zeros
from random import randrange
import pandas as pd
import sys
sys.path.append('..')
import configuration.model as config
from algorithm_extensions.mcc_score import mcc_score
from utils.common import get_timestamp


def save_record(df, index, params, mcc, fold):
    nan = -666
    kernel = params['kernel']
    c = params['C']
    class_weight = params['class_weight']
    gamma = nan
    if 'gamma' in params.keys():
        gamma = params['gamma']
    degree = nan
    if 'degree' in params.keys():
        degree = params['degree']
    coef0 = nan
    if 'coef0' in params.keys():
        coef0 = params['coef0']
    # updating data frame
    df[index] = (c, kernel, class_weight, gamma, degree, coef0, mcc, fold)


def train_and_validate(hyperparams_list):
    outer = config.number_of_cross_validation_parts
    data_yaml_scheme_path = config.data_yaml_scheme
    dataset_files = config.data_dict
    seed = config.seed
    data_format = [('c', 'f8'), ('kernel', 'a20'), ('class_weight', 'a5'), ('gamma', 'f8'), ('degree', 'i2'),
                   ('coef0', 'f8'), ('mcc', 'f8'), ('fold', 'i2')]
    outer_df_size = outer
    inner_df_size = len(hyperparams_list) * (outer_df_size-1) * outer

    outer_df = zeros(outer_df_size, dtype=data_format)
    inner_df = zeros(inner_df_size, dtype=data_format)

    inner_index = 0

    with open(data_yaml_scheme_path) as f:
            data_yaml_scheme = f.read()

    # OUTER LOOP
    for i in xrange(outer):
        print "#OUTER_LOOP:", i
        # TODO zobaczyc czy osie sa sensownie
        # TODO jakies testowanko, maybe?
        # we don't need to generate it right know. We'll do it after the inner loop to save RAM

        train_parts = [x for x in xrange(outer) if x != i]
        # we don't generate train set right now, we'll do it later, splitted to trtr and trte
        # we won't mix in the unlabelled samples, SVM will not like it

        for j in train_parts:
            # prepare datasets
            inner_train_parts = [x for x in train_parts if x != j]
            train_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                                    'y_val': dataset_files['labeled_values'],
                                                    'cv': [outer, inner_train_parts],
                                                    'seed': seed,
                                                    'middle_path': [],
                                                    'middle_val': []
                                                    }

            validation_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                                         'y_val': dataset_files['labeled_values'],
                                                         'cv': [outer, [j]],
                                                         'seed': seed,
                                                         'middle_path': [],
                                                         'middle_val': []
                                                         }

            train_data = yaml_parse.load(train_data_string)
            valid_data = yaml_parse.load(validation_data_string)

            # HYPERPARAMETER LOOP
            for hyperparams_dict in hyperparams_list:
                print '#STARTED procedure for:', hyperparams_dict, get_timestamp()
                # THE INNER LOOP

                # create model, learn it, check its prediction power on validation data
                classifier = svm.SVC(**hyperparams_dict)
                print 'starting training classifier', get_timestamp()
                classifier.fit(train_data.X, train_data.y.reshape(train_data.y.shape[0]))        # X, y
                print 'finished', get_timestamp()
                # calculate MCC
                print 'starting prediction phase', get_timestamp()
                predictions = classifier.predict(valid_data.X)    # returns numpy array
                print 'finished prediction phase', get_timestamp()
                mcc = mcc_score(true_y=valid_data.y.reshape(valid_data.y.shape[0]), predictions=predictions)

                # saving resutls
                print "#PARAMS:", hyperparams_dict
                print "#MCC SCORE:", mcc, '\n'
                save_record(inner_df, inner_index, hyperparams_dict, mcc, i)
                inner_index += 1
                # casting numpy array to data frame object
                df = pd.DataFrame(data=inner_df)
                # generating random name not to lost data in case of bad luck
                random_number = randrange(3)
                random_name = 'inner_data_frame_'+str(random_number)+'.csv'
                df.to_csv(random_name)

        # back to outer loop

        # do nothing, we'll do it later

        # # prepare testing set
        # outer_train_data_string = data_yaml_scheme % {
        #     'path': dataset_files['labeled_paths'],
        #     'y_val': dataset_files['labeled_values'],
        #     'cv': [outer, train_parts],
        #     'seed': seed,
        #     'middle_path': [],
        #     'middle_val': []
        #     }
        # test_data_string = data_yaml_scheme % {
        #     'path': dataset_files['labeled_paths'],
        #     'y_val': dataset_files['labeled_values'],
        #     'cv': [outer, [i]],
        #     'seed': seed,
        #     'middle_path': [],
        #     'middle_val': []
        #     }
        # outer_train_data = yaml_parse.load(outer_train_data_string)
        # test_data = yaml_parse.load(test_data_string)
        #
        # # get best from inner pandas TODO
        # # create pandas data frame
        # df = pd.DataFrame(data=inner_df)
        # df_fold = df['fold' == i]
        #
        # params = {}
        #
        # classifier = svm.SVC(**params)
        # classifier.fit(outer_train_data.X, outer_train_data.y.reshape(outer_train_data.y.shape[0]))
        # outer_predictions = classifier.predict(test_data.X)
        # outer_mcc = mcc_score(true_y=test_data.y.reshape(test_data.y.shape[0]), predictions=outer_predictions)
        # print "OUTER PARAMS:", params
        # print "OUTER MCC ON TEST TEST:", outer_mcc
        # sys.stdout.flush()
        #
        # # outer pandas TODO


# returns list of dictionaries that include named parameters for SVM constructors
def hyperparameters():
    max_iter = 4600*1000
    print 'PRODUCING HYPERPARAMETERS.'
    hyperparameters_list = []
    # grid search
    for c in [0.01, 0.1, 1, 10, 100, 1000]:     # we start with big c, because small c give poor performance anyway
        for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            if kernel == 'linear':
                hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto', 'max_iter': max_iter})
            if kernel == 'rbf':
                for gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                    hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto',
                                                 'gamma': gamma, 'max_iter': max_iter})
            if kernel == 'poly':
                for degree in [2, 3, 4, 5]:
                    for coef0 in [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100]:
                        hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto',
                                                     'degree': degree, 'coef0': coef0, 'max_iter': max_iter})
            if kernel == 'sigmoid':
                for coef0 in [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100]:
                    hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto',
                                                 'coef0': coef0, 'max_iter': max_iter})

    print 'DONE.'
    sys.stdout.flush()
    return hyperparameters_list


# main
def run():
    train_and_validate(hyperparameters())
