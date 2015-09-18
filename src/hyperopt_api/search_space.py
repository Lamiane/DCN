__author__ = 'agnieszka'

from hyperopt import hp

# TODO: This piece of code needs lots of documentation. ...
# TODO: ... Values that are set in here might be changed in parser.py, ...
# TODO: ... so the whole module MUST be well documented.


def get_search_space():

    space = [
        {
            'h0': hp.choice('first layer', [
                {
                    'h0 layer type': hp.choice('h0 layer type', ['RectifiedLinear', 'Sigmoid', 'Tanh']),
                    'h0 layer size multiplier': hp.choice('h0 layer size multiplier',
                                                          [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                },
            ]),
        },
        {
            'h1': hp.choice('second layer', [
                {
                    'h1 layer type': hp.choice('h1 layer type', ['RectifiedLinear', 'Sigmoid', 'Tanh']),
                    'h1 layer size multiplier': hp.choice('h1 layer size multiplier',
                                                          [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                },
                None
            ]),
        },
        {
            'h2': hp.choice('third layer', [
                {
                    'h2 layer type': hp.choice('h2 layer type', ['RectifiedLinear', 'Sigmoid', 'Tanh']),
                    'h2 layer size multiplier': hp.choice('h2 layer size multiplier',
                                                          [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                },
                None
            ]),
        }
        ]

    return space
