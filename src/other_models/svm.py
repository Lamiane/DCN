from sklearn import svm
import configuration.model as config
from pylearn2.config import yaml_parse
import sys
sys.path.append('..')
from algorithm_extensions.mcc_score import mcc_score


def train_and_validate(classifier):
    k = config.number_of_cross_validation_parts
    data_yaml_scheme_path = config.data_yaml_scheme
    dataset_files = config.data_dict
    seed = config.seed

    with open(data_yaml_scheme_path) as f:
            data_yaml_scheme = f.read()

    list_of_scores = []
    for i in xrange(k):
        validation_part = [i]
        test_part = [0]
        if i < k-1:
            test_part = [i+1]
        train_parts = [x for x in xrange(k) if x not in validation_part and x not in test_part]

        train_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                                'y_val': dataset_files['labeled_values'],
                                                'cv': [k, train_parts],
                                                'seed': seed,
                                                'middle_path': dataset_files['middle_paths'],
                                                'middle_val': dataset_files['middle_values']
                                                }

        # we don't want any unlabelled examples in validation nor test data
        validation_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                                     'y_val': dataset_files['labeled_values'],
                                                     'cv': [k, validation_part],
                                                     'seed': seed,
                                                     'middle_path': [],
                                                     'middle_val': []
                                                     }

        test_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                               'y_val': dataset_files['labeled_values'],
                                               'cv': [k, test_part],
                                               'seed': seed,
                                               'middle_path': [],
                                               'middle_val': []
                                               }

        # TODO zobaczyc czy osie sa sensownie
        # TODO co ja robie z valid_data, co ja robie ze swoim zyciem w ogole?
        # TODO jakies testowanko, maybe?
        train_data = yaml_parse.load(train_data_string)
        valid_data = yaml_parse.load(validation_data_string)
        test_data = yaml_parse.load(test_data_string)

        classifier.fit(train_data.topo_view, train_data.y)        # X, y
        # calculate MCC
        predictions = classifier.predict(test_data.topo_view)    # returns numpy array
        mcc = mcc_score(true_y=test_data.y, predictions=predictions)
        list_of_scores.append(mcc)
    print 'list of mcc scores for this architecture', list_of_scores

# grid search
for c in [0.01, 0.1, 1, 10, 100, 1000]:
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        if kernel == 'linear':
            print 'C:', c
            print 'kernel:', kernel
            classifier = svm.SVC(C=c, kernel=kernel, class_weights='auto')
            train_and_validate(classifier)
        if kernel == 'rbf':
            for gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                print 'C:', c
                print 'kernel:', kernel
                print 'gamma:', gamma
                classifier = svm.SVC(C=c, kernel=kernel, gamma=gamma, class_weights='auto')
                train_and_validate(classifier)
        if kernel == 'poly':
            for degree in [2, 3, 4, 5]:
                for coef0 in [-100, -10, -1, -0.1, -0.01, 0, 00.1, 0.1, 1, 10, 100]:
                    print 'C:', c
                    print 'kernel:', kernel
                    print 'degree:', degree
                    print 'coef0:', coef0
                    classifier = svm.SVC(C=c, kernel=kernel, degree=degree, coef0=coef0, class_weights='auto')
                    train_and_validate(classifier)
        if kernel == 'sigmoid':
            for coef0 in [-100, -10, -1, -0.1, -0.01, 0, 00.1, 0.1, 1, 10, 100]:
                print 'C:', c
                print 'kernel:', kernel
                print 'coef0:', coef0
                classifier = svm.SVC(C=c, kernel=kernel, coef0=coef0, class_weights='auto')
                train_and_validate(classifier)

