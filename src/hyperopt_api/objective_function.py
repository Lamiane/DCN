__author__ = 'agnieszka'
from blessings import Terminal
t = Terminal()
import sys
sys.path.append('..')
from utils.common import get_timestamp
import configuration.model as config
from training_algorithm.cross_validation import CrossValidator

def objective_function(samp):
    current_time = get_timestamp()
    k = config.number_of_cross_validation_parts
    yaml_scheme_path = config.yaml_skelton_path
    data_scheme_yaml = config.data_yaml_scheme
    dataset_dict = config.data_dict
    seed = config.seed
    return CrossValidator.run(k=k, model_dictionary=samp, data_yaml_scheme_path=data_scheme_yaml,
                              dataset_files=dataset_dict, seed=seed)


# based on pylearn2's scripts/plot_monitor.py


