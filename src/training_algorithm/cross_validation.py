__author__ = 'nex'
from os.path import join
from numpy import mean
from pylearn2.config import yaml_parse
import traceback
from blessings import Terminal
t = Terminal()
import sys
sys.path.append('..')
from hyperopt_api.parser import build
import configuration.model as config
from utils.common import get_timestamp
from yaml_parser import yaml_parser as yp
from algorithm_extensions.value_getters import f1_score_1threshold_get_value


class CrossValidator(object):

    def run(self, k, model_dictionary, model_yaml_scheme, data_yaml_scheme, dataset_files, seed=1337):
        # obtain the yaml skelton
        with open(config.yaml_skelton_path) as f:
            default_string = f.read()

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
                                                    'middle_val': dataset_files['middle_val']
                                                    }

            validation_data_string = data_yaml_scheme % {'path': dataset_files['labeled_paths'],
                                                         'y_val': dataset_files['labeled_values'],
                                                         'cv': [k, train_parts],
                                                         'seed': seed,
                                                         'middle_path': dataset_files['middle_paths'],
                                                         'middle_val': dataset_files['middle_val']
                                                         }

            mod = build(model_dictionary)   # based on description generated build an object that will fit into
                                            # yaml_parser
            print t.bold_cyan('SAMP'), model_dictionary
            print t.bold_blue('MODEL'), mod

            # define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
            weight_decay_coeffs = yp.parse_weight_decay(mod)

            # generate a filename to store the best model
            pkl_filename = join(config.path_for_storing, current_time + "_best.pkl")

            # create dictionary with hyper parameters
            hyper_params = {'model': yp.parse_to_yaml(mod),
                            'train_data': train_data_string,
                            'validation_data': validation_data_string,
                            'weight_decay_coeffs': weight_decay_coeffs,
                            'pkl_filename': pkl_filename
                            }
            yaml_string = default_string % hyper_params

            network = None
            # misclass_error = 1
            f1_score_error = 1

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
                        # misclass_error = lowest_misclass_error(network.model)
                        # f1_score_error = 1 - f1_score(network)
                        f1_score_error, threshold = f1_score_1threshold_get_value(network)
                        print t.bold_red("D_OF1: Best score for this model: "+str(f1_score_error))
                        print t.bold_red("D_OF1: Obtained for threshold: "+str(threshold))
                        f1_score_error = 1 - f1_score_error
                    except BaseException:  # TODO: this exception is to broad
                        with open(current_time + '_f1_error', 'w') as ERROR_FILE:
                            ERROR_FILE.write(traceback.format_exc())

                # print t.bold_red("M_01: misclass_error for this model: "+str(misclass_error))
                # return misclass_error
                print t.bold_red("M_02: f1 score error for this model: " + str(f1_score_error))
                list_of_scores.append(f1_score_error)

        # end of for
        return mean(list_of_scores)
