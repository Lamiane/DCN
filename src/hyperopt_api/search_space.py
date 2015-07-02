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
                    'layer type': 'ConvRectifiedLinear',
                    'output channels': hp.choice('output channels', [16, 32]),
                    'kernel shape width': hp.choice('kernel shape width', [6, 8, 10, 12]),
                    'kernel shape height': hp.choice('kernel shape height', [4, 5, 6, 7, 8]),   # TODO: moze jednak dorzucic 9 i 10
                    'kernel stride width': hp.choice('kernel stride width', [2, 4, 6]),
                    'kernel stride height': hp.choice('kernel stride height', [2, 3]),
                    'pool shape': hp.choice('pool shape', [(1, 1), (2, 1), (2, 2)]),
                    # choosing pool stride height equal or smaller than in pool shape
                    'pool stride height': hp.choice('pool stride height', [0.5, 1]),
                    # choosing pool stride width equal or bigger than the height
                    'pool stride width': hp.choice('pool stride width', [0.5, 1]),

                },
            ]),
        },
        {
            'h1': hp.choice('second layer', [
                {
                    'layer type': 'ConvRectifiedLinear',
                    'output channels': hp.choice('output channels', [16, 32]),
                    'kernel shape width': hp.choice('kernel shape width', [6, 8, 10, 12]),
                    'kernel shape height': hp.choice('kernel shape height', [4, 5, 6, 7, 8]),   # TODO: moze jednak dorzucic 9 i 10
                    'kernel stride width': hp.choice('kernel stride width', [2, 4, 6]),
                    'kernel stride height': hp.choice('kernel stride height', [2, 3]),
                    'pool shape': hp.choice('pool shape', [(1, 1), (2, 1), (2, 2)]),
                    # choosing height equal or smaller than in pool shape
                    'pool stride height': hp.choice('pool stride height', [0.5, 1]),
                    # choosing width equal or bigger than the height
                    'pool stride width': hp.choice('pool stride width', [0.5, 1]),

                },
                None
            ]),
        }
        ]

    return space





