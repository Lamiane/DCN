from sklearn import svm
from pylearn2.config import yaml_parse
from numpy import mean
import sys
sys.path.append('..')
import configuration.model as config
from algorithm_extensions.mcc_score import mcc_score


def train_and_validate(hyperparams_list):
    outer = config.number_of_cross_validation_parts
    data_yaml_scheme_path = config.data_yaml_scheme
    dataset_files = config.data_dict
    seed = config.seed

    with open(data_yaml_scheme_path) as f:
            data_yaml_scheme = f.read()

    # OUTER LOOP
    outer_list_of_scores = []
    mean_scores = []
    for i in xrange(outer):
        print "#OUTER_LOOP:", i
        # TODO zobaczyc czy osie sa sensownie
        # TODO jakies testowanko, maybe?
        # we don't need to generate it right know. We'll do it after the inner loop to save RAM

        train_parts = [x for x in xrange(outer) if x != i]
        # we don't generate train set right now, we'll do it later, splitted to trtr and trte
        # we won't mix in the unlabelled samples, SVM will not like it

        # HYPERPARAMETER LOOP
        for hyperparams_dict in hyperparams_list:
            # THE INNER LOOP
            inner_list_of_scores = []
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
                validation_data_string = {'path': dataset_files['labeled_paths'],
                                          'y_val': dataset_files['labeled_values'],
                                          'cv': [outer, [j]],
                                          'seed': seed,
                                          'middle_path': [],
                                          'middle_val': []
                                          }

                train_data = yaml_parse.load(train_data_string)
                valid_data = yaml_parse.load(validation_data_string)
                # TODO sprawdz, co siedzi w tym y, bo to moze byc jakas fancy rozszerzona reprezentacja
                import sys
                print train_data.y[1:5]
                sys.exit(0)

                # create model, learn it, check its prediction power on validation data
                classifier = svm.SVC(**hyperparams_dict)
                classifier.fit(train_data.topo_view, train_data.y)        # X, y
                # calculate MCC
                predictions = classifier.predict(valid_data.topo_view)    # returns numpy array
                mcc = mcc_score(true_y=valid_data.y, predictions=predictions)
                inner_list_of_scores.append(mcc)

            # saving resutls
            print "#PARAMS:", hyperparams_dict
            print "#SCORES:", inner_list_of_scores, '/n'
            mean_scores.append([hyperparams_dict, mean(inner_list_of_scores)])

        # back to outer loop
        # prepare testing set
        outer_train_data_string = data_yaml_scheme % {
            'path': dataset_files['labeled_paths'],
            'y_val': dataset_files['labeled_values'],
            'cv': [outer, train_parts],
            'seed': seed,
            'middle_path': [],
            'middle_val': []
            }
        test_data_string = data_yaml_scheme % {
            'path': dataset_files['labeled_paths'],
            'y_val': dataset_files['labeled_values'],
            'cv': [outer, [i]],
            'seed': seed,
            'middle_path': [],
            'middle_val': []
            }
        outer_train_data = yaml_parse.load(outer_train_data_string)
        test_data = yaml_parse.load(test_data_string)
        params, score = max(mean_scores, key=lambda l: l[1])
        classifier = svm.SVC(**params)
        classifier.fit(outer_train_data.topo_view, outer_train_data.y)
        outer_predictions = classifier.predict(test_data.topo_view)
        outer_mcc = mcc_score(true_y=test_data.y, predictions=outer_predictions)
        print "OUTER PARAMS:", params
        print "OUTER MCC ON TEST TEST:", outer_mcc
        outer_list_of_scores.append([params, outer_mcc])


# returns list of dictionaries that include named parameters for SVM constructors
def hyperparameters():
    print 'PRODUCING HYPERPARAMETERS.'
    hyperparameters_list = []
    # grid search
    for c in [0.01, 0.1, 1, 10, 100, 1000]:
        for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            if kernel == 'linear':
                hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weights': 'auto'})
            if kernel == 'rbf':
                for gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
                    hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weights': 'auto',
                                                 'gamma': gamma})
            if kernel == 'poly':
                for degree in [2, 3, 4, 5]:
                    for coef0 in [-100, -10, -1, -0.1, -0.01, 0, 00.1, 0.1, 1, 10, 100]:
                        hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weights': 'auto',
                                                     'degree': degree, 'coef0': coef0})
            if kernel == 'sigmoid':
                for coef0 in [-100, -10, -1, -0.1, -0.01, 0, 00.1, 0.1, 1, 10, 100]:
                    hyperparameters_list.append({'C': c, 'kernel': kernel, 'class_weights': 'auto',
                                                 'coef0': coef0})

    print 'DONE.'
    return hyperparameters_list

# main
if __name__ == "__main__":
    train_and_validate(hyperparameters())
