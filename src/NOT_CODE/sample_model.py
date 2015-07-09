__author__ = 'agnieszka'
import sys
sys.path.append('..')

import yaml_maker.yaml_parser as yp
from os.path import join
from pylearn2.config import yaml_parse
from tmp import notify


def test():
    # setting parameters of convolutional layer
    con = yp.ConvElemwise()
    con.layer_name = "con_layer"
    con.output_channels = 1
    con.kernel_shape = [2, 2]
    con.kernel_stride = [2, 2]
    con.nonlinearity = yp.TanhConvNonlinearity()
    con.irange = 0.1
    con.pool_shape = [1, 1]
    con.pool_stride = [1, 1]
    con.pool_type = 'max'

    # setting parameters of softmax layer
    sof = yp.Softmax()
    sof.n_classes = 3
    sof.irange = 0.1
    sof.layer_name = "softmax_layer"

    # creating list of layers
    layers = [con, sof]

    # creating space
    some_space = yp.Conv2DSpace()
    some_space.axes = ['c', 0, 1, 'b']
    some_space.shape = [6, 3474]    # TODO: brzydkie i inne niz u dra, ale inaczej nie dziala :(
    some_space.num_channels = 1

    # setting parameters of MLP model
    sett_model = yp.MLP()
    sett_model.layers = layers
    sett_model.input_space = some_space

    # creating a dictionary with hyperparameters
    dirpath = '../../data'
    path = [join(dirpath, '2RH1_actives_2dfp.dat'),
            join(dirpath, '2RH1_inactives_2dfp.dat'),
            join(dirpath, '2RH1_middle_2dfp.dat')]

    hyper_params = {'model': yp.parse_to_yaml(sett_model), 'path': yp.parse_to_yaml(path)}

    # obtaining the yaml skelton
    with open("example.yaml") as f:
        default_string = f.read()

    # filling the yaml skelton with hyperparameters
    yaml_string = default_string % hyper_params

    with open('generated_model.yaml', 'w') as g:
        g.write(yaml_string)

    # creating the model based on a yaml
    network = yaml_parse.load(yaml_string)

    # training the model
    print type(network)

    if hasattr(network, 'model'):
        print type(network.model)
        if hasattr(network.model, 'input_space'):
            print "network.model.input_space", network.model.input_space
        else:
            print "network model has not input space"

    if hasattr(network, 'dataset'):
        print type(network.dataset)

    print hasattr(network, 'input_space')
    if hasattr(network, 'input_space'):
        print network.input_space

    network.main_loop()

    notify()
