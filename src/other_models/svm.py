from sklearn import svm
from numpy import zeros
from random import randrange
import pandas as pd
import numpy as np
from os.path import join
from copy import deepcopy
import sys
sys.path.append('..')
import configuration.model as config
from algorithm_extensions.mcc_score import mcc_score
from utils.common import get_timestamp
from utils import values
from utils.casting import pred_and_trues_to_type_dict
from elm import load_data, divide_data
import data
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


def save_record(df, index, model_class, params, mcc, predictions_stats, outer_fold, inner_fold):
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

    tp = predictions_stats[values.TP]
    tn = predictions_stats[values.TN]
    fp = predictions_stats[values.FP]
    fn = predictions_stats[values.FN]

    df[index] = (model_class, c, kernel, class_weight, gamma, degree, coef0,
                 tp, tn, fp, fn, outer_fold, inner_fold, mcc)


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
    experiment_name = 'svms'

    data_format = [('model_name', 'a40'), ('c', 'f8'), ('kernel', 'a20'), ('class_weight', 'a5'),
                   ('gamma', 'f8'), ('degree', 'i2'), ('coef0', 'f8'),
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
                                                 'gamma': gamma, 'max_iter': max_iter, 'model_class': svm})
            if kernel == 'poly':
                for degree in [2, 3, 4, 5]:
                    for coef0 in [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100]:
                        hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto',
                                                     'degree': degree, 'coef0': coef0, 'max_iter': max_iter,
                                                     'model_class': svm})
            if kernel == 'sigmoid':
                for coef0 in [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100]:
                    hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weight': 'auto',
                                                 'coef0': coef0, 'max_iter': max_iter, 'model_class': svm})

    print 'DONE.'
    sys.stdout.flush()
    return hyperparameters_list


if __name__ == '__main__':
    import sys
    # fold value should be in range 0...n_of_outer_folds-1

    fold = int(sys.argv[1])
    train_and_validate(fold, hyperparameters())
