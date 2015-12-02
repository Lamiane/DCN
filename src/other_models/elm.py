from pylearn2.config import yaml_parse
from numpy import zeros
from random import randrange
import pandas as pd
import sys
sys.path.append('..')
import configuration.model as config
from algorithm_extensions.mcc_score import mcc_score
from utils.common import get_timestamp
from utils import values
from utils.casting import pred_and_trues_to_type_dict


def save_record(df, index, params, mcc, predictions_stats, outer_fold, inner_fold):
    h = params['h']
    c = params['C']
    f = 'tanimoto'
    balanced = params['balanced']
    tp = predictions_stats[values.TP]
    tn = predictions_stats[values.TN]
    fp = predictions_stats[values.FP]
    fn = predictions_stats[values.FN]

    # updating data frame
    df[index] = (c, h, f, balanced, tp, tn, fp, fn, outer_fold, inner_fold, mcc)


def train_and_validate(hyperparams_list):
    outer = config.number_of_cross_validation_parts
    data_yaml_scheme_path = config.data_yaml_scheme
    dataset_files = config.data_dict
    seed = config.seed
    data_format = [('c', 'f8'), ('h', 'i1'), ('f', 'a20'), ('balanced', 'a6'),
                   ('TP', 'i2'), ('TN', 'i2'), ('FP', 'i2'), ('FN', 'i2'),
                   ('outer_fold', 'i2'), ('inner_fold', 'i2'), ('mcc', 'f8')]
    outer_df_size = outer
    inner_df_size = len(hyperparams_list) * (outer_df_size-1)

    outer_df = zeros(outer_df_size, dtype=data_format)
    inner_df = zeros(inner_df_size, dtype=data_format)

    inner_index = 0

    with open(data_yaml_scheme_path) as f:
            data_yaml_scheme = f.read()

    # OUTER LOOP
    for i in xrange(outer):
        print "#OUTER_LOOP:", i
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

            # y must satisfy ELM's class assumptions
            train_data.y[train_data.y == 0] = -1    # np.array so changes in place, no copy created
            valid_data.y[valid_data.y == 0] = -1    # no need to assign anywere

            # HYPERPARAMETER LOOP
            for hyperparams_dict in hyperparams_list:
                print '#STARTED procedure for:', hyperparams_dict, get_timestamp()
                # THE INNER LOOP

                # create model, learn it, check its prediction power on validation data
                classifier = XELM(**hyperparams_dict)
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
                prediction_stats = pred_and_trues_to_type_dict(valid_data.y.reshape(valid_data.y.shape[0]), predictions)
                save_record(inner_df, inner_index, hyperparams_dict, mcc, prediction_stats, i, j)
                inner_index += 1
                # casting numpy array to data frame object
                df = pd.DataFrame(data=inner_df)
                # generating random name not to lost data in case of bad luck
                random_number = randrange(3)
                random_name = 'XELM_inner_data_frame_'+str(random_number)+'.csv'
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
    print 'PRODUCING HYPERPARAMETERS.'
    hyperparameters_list = []
    # grid search
    for c in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        for h in [1, 2, 3, 4, 5]:
                hyperparameters_list.append({'C': c, 'h': h, 'balanced': 'True'})

    print 'DONE. LENGTH:', len(hyperparameters_list)
    sys.stdout.flush()
    return hyperparameters_list


# main
def run():
    train_and_validate(hyperparameters())


# author: W. M. Czarnecki
"""
Implementation assumes that there are TWO LABELS, namely -1 and +1.
If you have different labels you have to preproess them. Furthermore
be sure to correctly set "balanced" hyperparameter accordingly to the
metric you want to optimize
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csr_matrix


def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX+WW-XW)


class ELM(object):
    """ Extreme Learning Machine """

    def __init__(self, h, C=10000, f=tanimoto, random_state=666, balanced=False):
        """
        h - number of hidden units
        C - regularization strength (L2 norm)
        f - activation function [default: tanimoto]
        balanced - if set to true, model with maximize GMean (or Balanced accuracy),
                   if set to false [default] - model will maximize Accuracy
        """
        self.h = h
        self.C = C
        self.f = f
        self.rs = random_state
        self.balanced = balanced

    def _hidden_init(self, X, y):
        """ Initializes hidden layer """
        np.random.seed(self.rs)
        W = csr_matrix(np.random.rand(self.h, X.shape[1]))
        b = np.random.normal(size=self.h)
        return W, b

    def fit(self, X, y):
        """ Fits ELM to training samples X and labels y """
        self.W, self.b = self._hidden_init(X, y)
        H = self.f(X, self.W, self.b)

        if self.balanced:
            counts = { l : float(y.tolist().count(l)) for l in set(y) }
            ms = max([ counts[k] for k in counts ])
            self.counts = { l : np.sqrt( ms/counts[l] ) for l in counts }
        else:
            self.counts = { l: 1 for l in set(y) }

        w = np.array( [[ self.counts[a] for a in y ]] ).T
        H = np.multiply(H, w)
        y = np.multiply(y.reshape(-1,1), w).ravel()

        self.beta = la.inv(H.T.dot(H) + 1.0 / self.C * np.eye(H.shape[1])).dot((H.T.dot(y)).T)

    def predict(self, X):
        H = self.f(X, self.W, self.b)
        return np.array(np.sign(H.dot(self.beta)).tolist())


class XELM(ELM):
    """ Extreme Learning Machine initialized with training samples """

    def _hidden_init(self, X, y):

        h = min(self.h, X.shape[0]) # hidden neurons count can't exceed training set size

        np.random.seed(self.rs)
        W = X[np.random.choice(range(X.shape[0]), size=h, replace=False)]
        b = np.random.normal(size=h)
        return W, b


class TWELM(XELM):
    """
    TWELM* model from
    "Weighted Tanimoto Extreme Learning Machine with case study of Drug Discovery"
    WM Czarnecki, IEEE Computational Intelligence Magazine, 2015
    """
    def __init__(self, h, C=10000, random_state=666):
        super(TWELM, self).__init__(h=h, C=C, f=tanimoto, random_state=random_state, balanced=True)