__author__ = 'agnieszka'


def run(max_evals=10):
    from hyperopt import fmin, tpe
    import sys
    sys.path.append('..')
    from search_space import get_search_space
    from objective_function import objective_function
    best = fmin(objective_function, get_search_space(), algo=tpe.suggest, max_evals=max_evals)
    print "BEST MODEL"
    print best
    return best


def run_no_hyperopt():
    import sys
    sys.path.append('..')
    import configuration.model as config
    from training_algorithm.cross_validation import CrossValidator
    k = config.number_of_cross_validation_parts
    data_scheme_yaml = config.data_yaml_scheme
    dataset_dict = config.data_dict
    seed = config.seed
    return CrossValidator.run(k=k, data_yaml_scheme_path=data_scheme_yaml, dataset_files=dataset_dict, seed=seed)
