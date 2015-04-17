from operator import mul
from numpy import zeros, nditer, size
from random import choice


# tak na prawde prostokat
# TODO: dodac parametr distribution
# TODO: vector bedzie TensorVariablem, trzeba sobie z tym poradzic
# wyglada na to, ze dziala
# def define_squerization_map(vector, square_size):
#     number_of_elements = reduce(mul, square_size)
#     if number_of_elements != len(vector):
#         pass  # raise some error
#     sq_map = zeros(square_size, dtype='int')
#     indices = list(xrange(number_of_elements))
#
#     for el in nditer(sq_map, op_flags=['readwrite']):
#         el[...] = choice(indices)
#         indices.remove(el[...])
#
#     return sq_map


# o zadanej wielkosci
# TODO: dodac distribution
# in:
# square_size - a python list with dimensions
# out:
# numpy.ndarray
def create_squerization_map(square_size):
    number_of_elements = reduce(mul, square_size)
    sq_map = zeros(square_size, dtype='int')
    indices = list(xrange(number_of_elements))

    for el in nditer(sq_map, op_flags=['readwrite']):
        el[...] = choice(indices)
        indices.remove(el[...])

    return sq_map

# wyglada na to ze dziala
# in:
# vector - theano variable (1D tensor)
# squerization_map - numpy.ndarray obtainable from create_squerization_map
# out:
# numpy.ndarray
def squerize(vector, squerization_map):
    if size(vector) != size(squerization_map):
        pass  # raise some error
    sq = zeros(squerization_map.shape, dtype="int")
    for i in xrange(squerization_map.shape[0]):
        for j in xrange(squerization_map.shape[1]):
            # pass
            sq[i][j] = vector.get_value()[squerization_map[i][j]]
    return sq