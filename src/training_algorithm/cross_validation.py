__author__ = 'nex'
from os.path import join
from numpy import mean
from pylearn2.config import yaml_parse
import traceback
from pylearn2.utils import serial
from blessings import Terminal
t = Terminal()
import sys
sys.path.append('..')
from hyperopt_api.parser import build_multilayer
import configuration.model as config
from utils.common import get_timestamp
from yaml_parser import yaml_parser as yp
from algorithm_extensions.get_predictions import Predictor


class CrossValidator(object):
    @staticmethod
    def run(k, data_yaml_scheme_path, dataset_files, seed=1337):
        print 'CV_ENTER'
        # obtain the yaml skelton
        assert k >= 3   # we need to have at least 3 sets: train, validation, test

        with open(config.yaml_skelton_path) as f:
            default_string = f.read()

        with open(data_yaml_scheme_path) as f:
            data_yaml_scheme = f.read()

        list_of_scores = []
        for i in xrange(k):
            current_time = get_timestamp()

            # calculate which parts to choose
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

            # mod = build_multilayer(model_dictionary)    # based on description generated build an object that
                                                        #  will fit into yaml_parser
            # print t.bold_cyan('SAMP'), model_dictionary
            # print t.bold_blue('MODEL'), mod

            # define weight decay params, which depend on the number of layers (there is one parameter for each layer)
            # weight_decay_coeffs = yp.general_parse_weight_decay(mod)

            # generate a filename to store the best model
            pkl_filename = join(config.path_for_storing, current_time + "_best.pkl")

            # multilayer = yp.parse_to_yaml(mod)

            # create dictionary with hyper parameters
            hyper_params = {
                            'train_data': train_data_string,
                            'validation_data': validation_data_string,
                            'pkl_filename': pkl_filename
                            }
            yaml_string = default_string % hyper_params

            network = None
            # misclass_error = 1
            # f1_score_error = 1
            roc_score = 0

            try:
                # create the model based on a yaml
                network = yaml_parse.load(yaml_string)
                print t.bold_magenta('NETWORK'), type(network)
                # train the model
                network.main_loop()

            except BaseException:  # TODO: this exception is to broad
                # if exception was thrown save yaml of a model that generated that exception
                with open(current_time + '.yaml', 'w') as YAML_FILE:
                    YAML_FILE.write(yaml_string)
                # write down errors description to a file
                with open(current_time + '.error', 'w') as ERROR_FILE:
                    ERROR_FILE.write(traceback.format_exc())

            finally:
                if network is not None:
                    try:
                        from numpy import argmax
                        # run predictions to obtain score for this model
                        test_data = yaml_parse.load(test_data_string)
                        best_model = serial.load('best_model_roc_youden.model')

                        predictor = Predictor(best_model)
                        predictions = predictor.get_predictions(test_data.X)

                        fp = 0
                        fn = 0
                        tp = 0
                        tn = 0

                        for tr, pre in zip(test_data.y, predictions):
                            if pre[0][1] > 0.5:
                                if tr[0] == 1:
                                    tp += 1
                                else:
                                    fp += 1
                            else:
                                if tr[0] == 0:
                                    tn += 1
                                else:
                                    fn += 1
                        roc_score = (float(tp)/(float(tp) + fn)) - (float(fp)/(float(fp) + tn))
                        list_of_scores.append(1-roc_score)  # we want to maximise this score, hyperopt minimises

                        print t.bold_red("_ROC: Best roc score for this model: "+str(roc_score))
                        precision = float(tp)/(tp + fp)
                        recall = float(tp)/(tp + fn)
                        f1score = 0
                        if precision+recall != 0:
                            f1score = 2*precision*recall/(precision+recall)
                        print 'precision:', precision
                        print "recall:", recall
                        print "f1score", f1score

                    except BaseException:  # TODO: this exception is to broad
                            with open(current_time + '_ROC_error', 'w') as ERROR_FILE:
                                ERROR_FILE.write(traceback.format_exc())

                # print t.bold_red("M_01: misclass_error for this model: "+str(misclass_error))
                # return misclass_error
                print t.bold_red("M_02: roc score error for this model: " + str(roc_score))
                # list_of_scores.append(f1_score_error)
        m = mean(list_of_scores)
        print "CV_MEAN mean(1-ROC) on this architecture:", m
        return m
