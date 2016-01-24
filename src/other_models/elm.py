from numpy import zeros
from random import randrange
import pandas as pd
import numpy as np
import sys
import math
from sklearn.utils import shuffle
from os.path import join
from scipy.sparse import csr_matrix
from copy import deepcopy
sys.path.append('..')
sys.path.append('/lhome/home/pocha/libs/anaconda/lib/python2.7')
sys.path.append('/lhome/home/pocha/libs/anaconda/lib/python2.7/site-packages')
# for blessings
sys.path.append('/lhome/home/pocha/libs/blessings-1.6')
# for bson (hyperopt need it)
sys.path.append('/lhome/home/pocha/libs/pymongo-3.0.3')
# for hyperopt
sys.path.append('/lhome/home/pocha/libs/hyperopt')
# for pylearn2
sys.path.append('/lhome/home/pocha/libs/pylearn2')
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

    balanced = params['balanced'] if 'balanced' in params else 'nan'
    random_state = params['random_state'] if 'random_state' in params else -666

    tp = predictions_stats[values.TP]
    tn = predictions_stats[values.TN]
    fp = predictions_stats[values.FP]
    fn = predictions_stats[values.FN]

    # updating data frame
    df[index] = (model_name, c, h, f, balanced, random_state, tp, tn, fp, fn, outer_fold, inner_fold, mcc)


def train_and_validate(fold_n, hyperparams_list):
    today = get_timestamp()[0:10]
    hyp_copy = deepcopy(hyperparams_list)
    #################################################
    # # # ! ! ! C O N F I G U R A T I O N ! ! ! # # #
    #################################################
    outer = config.number_of_cross_validation_parts_outer
    inner = config.number_of_cross_validation_parts_inner
    store_path = config.store_path
    # also config.actives_path, config_nonactives_path
    experiment_name = 'elms'

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
    tr_X, tr_y, te_X, te_y = load_data(outer, fold_n, config.actives_path, config.nonactives_path)

    print "#OUTER_LOOP:", fold_n
    # we don't need to generate it right know. We'll do it after the inner loop to save RAM

    for j in xrange(inner):
        ttrain_X, ttest_X = divide_data(tr_X, inner, j)
        ttrain_y, ttest_y = divide_data(tr_y, inner, j)

        # elms require sparse matrices
        ttrain_X = csr_matrix(ttrain_X)
        ttest_X = csr_matrix(ttest_X)

        # y-greks must be 2-dimensional
        # ttrain_y = np.reshape(a=ttrain_y, newshape=(ttrain_y.shape[0], 1))
        ttest_y = np.reshape(a=ttest_y, newshape=(ttest_y.shape[0], 1))

        # HYPERPARAMETER LOOP
        hyperparams_list = deepcopy(hyp_copy)
        for hyperparams_dict in hyperparams_list:
            print '#STARTED procedure for:', hyperparams_dict, get_timestamp()
            # THE INNER LOOP

            # create model, learn it, check its prediction power on validation data
            model_class = hyperparams_dict.pop('model_class')
            classifier = model_class(**hyperparams_dict)
            try:
                print 'starting training classifier', get_timestamp()
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
            except ValueError as ve:
                print ve
                # raise ve
                save_record(inner_df, inner_index, model_class, hyperparams_dict, -666,
                            {values.TP: -666, values.TN: -666, values.FP: -666, values.FN: -666},
                            fold_n, j)
            inner_index += 1
            # casting numpy array to data frame object
            df = pd.DataFrame(data=inner_df)
            # generating random name not to lost data in case of bad luck
            random_number = randrange(3)
            random_name = experiment_name+'_'+str(fold_n)+'_'+str(outer)+'_'+today+'_'+str(random_number)+'.csv'
            df.to_csv(join(store_path, random_name))

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
    actives_all = data.load(active_path)
    nonactives_all = data.load(nonactive_path)

    act_train, act_test = divide_data(actives_all, n_folds, test_fold)
    nact_train, nact_test = divide_data(nonactives_all, n_folds, test_fold)

    X_tr, y_tr = glue_and_generate_labels(act_train, nact_train)
    X_te, y_te = glue_and_generate_labels(act_test, nact_test)

    return X_tr, y_tr, X_te, y_te


def divide_data(array, n_folds, test_fold):
    fold_size = int(math.ceil(float(array.shape[0])/n_folds))
    test_fold_start = test_fold * fold_size
    test_fold_end = (test_fold + 1) * fold_size

    train = np.delete(array, range(test_fold_start, test_fold_end), 0)
    test = array[test_fold_start:test_fold_end]

    assert array.shape[0] == train.shape[0] + test.shape[0]
    if len(array.shape) == 2:
        assert array.shape[1] == train.shape[1]
        assert array.shape[1] == test.shape[1]

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
                for model_class in [ELM, XELM]:
                    hyperparameters_list.append({'C': c, 'h': h, 'balanced': balanced,
                                                 'model_class': model_class})

    for c in [1, 10, 100, 1000, 10000, 100000]:
        for h in [1, 2, 3, 4, 5]:
            for random_state in xrange(5):
                for model_class in [TWELM]:
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
