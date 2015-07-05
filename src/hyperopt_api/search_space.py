__author__ = 'agnieszka'

from hyperopt import hp

''' USEFUL INFORMATION
    Pylearn2 pool shape tuple is (width, height)
'''

# TODO: This piece of code needs a lot of documentation. ...
# TODO: ... Values that are set in here might be changed in parser.py, ...
# TODO: ... so the whole module MUST be well documented.


def get_search_space():

    space = [
        {
            'h0': hp.choice('first layer', [
                {
                    'h0 layer type': 'ConvRectifiedLinear',
                    'h0 output channels': hp.choice('h0 output channels', [16, 32]),
                    'h0 kernel shape width': hp.choice('h0 kernel shape width', [6, 8, 10, 12]),
                    'h0 kernel shape height': hp.choice('h0 kernel shape height', [4, 5, 6, 7, 8]),   # TODO: moze jednak dorzucic 9 i 10
                    'h0 kernel stride width': hp.choice('h0 kernel stride width', [2, 4, 6]),
                    'h0 kernel stride height': hp.choice('h0 kernel stride height', [2, 3]),
                    'h0 pool shape': hp.choice('h0 pool shape', [(1, 1), (2, 1), (2, 2)]),
                    # choosing pool stride height equal or smaller than in pool shape
                    'h0 pool stride height': hp.choice('h0 pool stride height', [0.5, 1]),
                    # choosing pool stride width equal or bigger than the height
                    'h0 pool stride width': hp.choice('h0 pool stride width', [0.5, 1]),

                },
            ]),
        },
        {
            'h1': hp.choice('second layer', [
                {
                    'h1 layer type': 'ConvRectifiedLinear',
                    'h1 output channels': hp.choice('h1 output channels', [16, 32]),
                    'h1 kernel shape width': hp.choice('h1 kernel shape width', [6, 8, 10, 12]),
                    'h1 kernel shape height': hp.choice('h1 kernel shape height', [4, 5, 6, 7, 8]),   # TODO: moze jednak dorzucic 9 i 10
                    'h1 kernel stride width': hp.choice('h1 kernel stride width', [2, 4, 6]),
                    'h1 kernel stride height': hp.choice('h1 kernel stride height', [2, 3]),
                    'h1 pool shape': hp.choice('h1 pool shape', [(1, 1), (2, 1), (2, 2)]),
                    # choosing height equal or smaller than in pool shape
                    'h1 pool stride height': hp.choice('h1 pool stride height', [0.5, 1]),
                    # choosing width equal or bigger than the height
                    'h1 pool stride width': hp.choice('h1 pool stride width', [0.5, 1]),

                },
                None
            ]),
        }
        ]

    return space





