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



