from pylearn2.config import yaml_parse
from numpy import zeros, nan
from random import randrange
import pandas as pd
import traceback
import sys
import numpy as np
import pickle as pkl
import math
from sklearn.utils import shuffle
sys.path.append('..')
import configuration.model as config
from algorithm_extensions.mcc_score import mcc_score
from utils.common import get_timestamp
from utils import values
from utils.casting import pred_and_trues_to_type_dict
from wojciech.elms import ELM, XELM, TWELM
import data


def save_record(df, index, model_class, params, mcc, predictions_stats, outer_fold, inner_fold):
    model_name = str(model_class)
    h = params['h']
    c = params['C']
    f = 'tanimoto'

    balanced = params['balanced'] if 'balanced' in params else nan     # todo check
    random_state = params['random_state'] if 'random_state' in params else nan     # todo check

    tp = predictions_stats[values.TP]
    tn = predictions_stats[values.TN]
    fp = predictions_stats[values.FP]
    fn = predictions_stats[values.FN]

    # updating data frame
    df[index] = (model_name, c, h, f, balanced, random_state, tp, tn, fp, fn, outer_fold, inner_fold, mcc)


def train_and_validate(fold_n, hyperparams_list):
    today = get_timestamp()[0:14]
    #################################################
    # # # ! ! ! C O N F I G U R A T I O N ! ! ! # # #
    #################################################
    outer = config.number_of_cross_validation_parts_outer
    inner = config.number_of_cross_validation_parts_inner
    # also config.actives_path, config_nonactives_path

    data_format = [('model_name', 'a40'), ('c', 'f8'), ('h', 'i1'), ('f', 'a20'),
                   ('balanced', 'a6'), ('random_state', 'i4'),
                   ('TP', 'i2'), ('TN', 'i2'), ('FP', 'i2'), ('FN', 'i2'),
                   ('outer_fold', 'i2'), ('inner_fold', 'i2'), ('mcc', 'f8')]
    outer_df_size = outer
    inner_df_size = len(hyperparams_list) * inner

    outer_df = zeros(outer_df_size, dtype=data_format)
    inner_df = zeros(inner_df_size, dtype=data_format)

    inner_index = 0

    # OUTER LOOP
    print 61, type(fold_n)
    tr_X, tr_y, te_X, te_y = load_data(outer, fold_n, config.actives_path, config.nonactives_path)
    print 63, tr_X.shape
    print 64, tr_y.shape
    print 65, te_X.shape
    print 66, te_y.shape

    print "#OUTER_LOOP:", fold_n
    # we don't need to generate it right know. We'll do it after the inner loop to save RAM

    for j in xrange(inner):
        ttrain_X, ttest_X = divide_data(tr_X, inner, j)
        ttrain_y, ttest_y = divide_data(tr_y, inner, j)

        # HYPERPARAMETER LOOP
        for hyperparams_dict in hyperparams_list:
            print '#STARTED procedure for:', hyperparams_dict, get_timestamp()
            # THE INNER LOOP

            # create model, learn it, check its prediction power on validation data
            model_class = hyperparams_dict.pop('model_class')
            classifier = model_class(**hyperparams_dict)
            try:
                print 'starting training classifier', get_timestamp()
                print 85, type(ttrain_X)
                classifier.fit(ttrain_X, ttrain_y)        # X, y
                print 'finished', get_timestamp()
                # calculate MCC
                print 'starting prediction phase', get_timestamp()
                predictions = classifier.predict(ttest_X)    # returns numpy array
                print 'finished prediction phase', get_timestamp()

                mcc = mcc_score(true_y=ttest_y, predictions=predictions)
                print "#MCC SCORE:", mcc, '\n'

                # saving resutls
                prediction_stats = pred_and_trues_to_type_dict(ttest_y, predictions)
                save_record(inner_df, inner_index, model_class, hyperparams_dict, mcc, prediction_stats,
                            fold_n, j)
            # except ValueError as ve:
            except KeyboardInterrupt:
                print ve
                raise ve
                save_record(inner_df, inner_index, model_class, hyperparams_dict, np.nan,
                            {values.TP: np.nan, values.TN: np.nan, values.FP: np.nan, values.FN: np.nan},
                            fold_n, j) # we will put mask later on
            inner_index += 1
            # casting numpy array to data frame object
            df = pd.DataFrame(data=inner_df)
            # generating random name not to lost data in case of bad luck
            random_number = randrange(3)
            random_name = 'elms_'+str(fold_n)+'_'+str(outer)+'_'+str(random_number)+today+'.csv'
            df.to_csv(random_name)

        # back to outer loop

        # do nothing, we'll do it later

        # testing set is te_X, te_y
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


def load_data(n_folds, test_fold, active_path, nonactive_path):
    print 131, type(test_fold)
    actives_all = data.load(active_path)
    nonactives_all = data.load(nonactive_path)

    act_train, act_test = divide_data(actives_all, n_folds, test_fold)
    nact_train, nact_test = divide_data(nonactives_all, n_folds, test_fold)

    X_tr, y_tr = glue_and_generate_labels(act_train, nact_train)
    X_te, y_te = glue_and_generate_labels(act_test, nact_test)

    return X_tr, y_tr, X_te, y_te


# TODO test this method!
def divide_data(array, n_folds, test_fold):
    print 146, type(test_fold)
    fold_size = int(math.ceil(float(array.shape[0])/n_folds))
    test_fold_start = test_fold * fold_size
    test_fold_end = (test_fold + 1) * fold_size

    train = np.delete(array, range(test_fold_start, test_fold_end), 0)
    test = array[test_fold_start:test_fold_end]

    print 163, type(train)
    print 164, type(test)

    return train, test


def glue_and_generate_labels(act, nact):
    act_y = np.ones(act.shape[0])
    nact_y = -np.ones(nact.shape[0])

    tr_X = np.vstack((act, nact))
    tr_y = np.hstack((act_y, nact_y))

    seed = 666
    X = shuffle(tr_X, random_state=seed)
    y = shuffle(tr_y, random_state=seed)

    return X, y


# returns list of dictionaries that include named parameters for SVM constructors
def hyperparameters():
    print 'PRODUCING HYPERPARAMETERS.'
    hyperparameters_list = []
    # grid search
    for c in [1, 10, 100, 1000, 10000, 100000]:
        for h in [1, 2, 3, 4, 5]:
            for balanced in ['True', 'False']:
                for model_class in [ELM, XELM]:     # TODO check
                    hyperparameters_list.append({'C': c, 'h': h, 'balanced': balanced,
                                                 'model_class': model_class})

    for c in [1, 10, 100, 1000, 10000, 100000]:
        for h in [1, 2, 3, 4, 5]:
            for random_state in xrange(5):
                for model_class in [TWELM]:   # TODO check
                    hyperparameters_list.append({'C': c, 'h': h, 'random_state': random_state,
                                                 'model_class': model_class})

    print 'DONE. LENGTH:', len(hyperparameters_list)
    sys.stdout.flush()
    return hyperparameters_list

if __name__ == '__main__':
    import sys
    # fold value should be in range 0...n_of_outer_folds-1

    fold = int(sys.argv[1])
    train_and_validate(fold, hyperparameters())
