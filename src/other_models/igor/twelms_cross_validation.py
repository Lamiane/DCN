import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, matthews_corrcoef, make_scorer
from scipy.sparse import csr_matrix
import argparse
import pickle as pkl
import sys
sys.path.append('..')
sys.path.append('../..')
import data
from utils.common import get_timestamp
from wojciech.sklearn_elms import ELM, TWELM, XELM

#################################################
# # # ! ! ! C O N F I G U R A T I O N ! ! ! # # #
#################################################
classifier = TWELM()
classifier_name = 'nopenope'
date = get_timestamp()[0:10]    # YYYY-MM-DD


def select_and_evaluate(data, parameters_file, cv_no=5, cv_repeats=1, tcv_no=5, n_jobs=1, random_seed=None):
    X, y = data
    X = csr_matrix(X)

    param_grid = pkl.load(open(parameters_file))

    # tablice rezultatow i slownik parametrow
    results = np.zeros((cv_no, 5))

    auc_scorer = make_scorer(matthews_corrcoef)
    # podzial na foldy
    skf = StratifiedKFold(y=y, n_folds=cv_no, shuffle=True, random_state=random_seed)
    for fold, (train_ind, test_ind) in enumerate(skf):
        # fold to jest jakis numerek inny dla kazdego folda, przyda sie do nazywania
        x_train = X[train_ind]
        y_train = y[train_ind]
        x_test = X[test_ind]
        y_test = y[test_ind]

        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=param_grid,
                                   scoring=auc_scorer,
                                   n_jobs=n_jobs,
                                   cv=tcv_no,
                                   refit=True,
                                   verbose=9)
        grid_search.fit(x_train, y_train)

        # saving grid scores
        grid_scores_filename = date+'_'+classifier_name+'_'+str(fold)+'_grid_scores.pkl'
        with open(grid_scores_filename, 'w') as gs:
            pkl.dump(grid_search.grid_scores_, gs)

        # saving best_estimator, its params, best_score
        best_filename = date+'_'+classifier_name+'_'+str(fold)+'_best_estimator_params_score.pkl'
        with open(best_filename, 'w') as bf:
            best = (grid_search.best_estimator_, grid_search.best_estimator_.get_params(), grid_search.best_score_)
            pkl.dump(best, bf)

        # saving best_params
        best_params_filename = date+'_'+classifier_name+'_'+str(fold)+'_best_params.pkl'
        with open(best_params_filename, 'w') as bp:
            pkl.dump(grid_search.best_params_, bp)

        # przelicz predykcje najlepszego modelu na zbiorze (x_test, y_test)
        predictions = grid_search.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_true=y_test, y_pred=predictions)
        # auc = roc_auc_score(y_true=y_test, y_score=grid_search.predict_proba(x_test)[:, 1])
        mth = matthews_corrcoef(y_true=y_test, y_pred=predictions)
        results[fold, :] = (acc, prec, rec, f1, mth)

        print "Fold {} results: ".format(fold)
        print 'Accuracy:        ', acc
        print 'Precision:       ', prec
        print 'Recall:          ', rec
        print 'f1_score:        ', f1
        # print 'roc_auc:         ', auc
        print 'Matthew\'s corr: ', mth
        print 'grid_scores_:    '

    # saving "results" array
    results_filename = date+'_'+classifier_name+'_'+'results.pkl'
    with open(results_filename, 'w') as res:
        pkl.dump(results, res)

    np.set_printoptions(precision=3, suppress=True)
    res = np.mean(results, 0)
    # print "accu\t{}  prec\t{}  recl\t{}  f1  \t{}  auc \t{}  mthw\t{}".format(res[0], res[1], res[2], res[3], res[4], res[5])
    print "accu\t{}  prec\t{}  recl\t{}  f1  \t{}  mthw\t{}".format(res[0], res[1], res[2], res[3], res[4])


if __name__ == '__main__':
    prsr = argparse.ArgumentParser()
    prsr.add_argument("--act", dest='actives', help='data file with active examples')
    prsr.add_argument("--nonact", dest='nonactives', help='data file with inactive examples')
    prsr.add_argument('--nfolds', dest='folds_no', type=int, default=5, help='number of folds [5]')
    prsr.add_argument('--seed', dest='seed', type=int, default=15, help='seed for the StratifiedKFold generator [15]')
    prsr.add_argument('--parameters', dest='parameters', default=None,
                      help='pickle file with grid search parameters [None]')
    prsr.add_argument('--njobs', dest='n_jobs', type=int, default=1, help='number of jobs to run concurrently')
    prsr.add_argument('--filename_prefix', dest='filename_prefix', help='prefix for generated filenames')

    args = prsr.parse_args()
    print args

    x_y = data.load_data(active_path=args.actives, nonactive_path=args.nonactives)
    cv_no = args.folds_no
    seed = args.seed
    parameters = args.parameters
    n_jobs = args.n_jobs

    classifier_name = args.filename_prefix

    select_and_evaluate(data=x_y, parameters_file=parameters, cv_no=cv_no, tcv_no=cv_no, n_jobs=n_jobs, random_seed=seed)
