__author__ = 'agnieszka'

from math import floor
from blessings import Terminal
import sys
sys.path.append('..')
from yaml_parser import models
import configuration.model as config
t = Terminal()


# returns object that can be parsed to yaml
def build(hyperopt_sample):

    initial_data_height = config.data_height
    initial_data_width = config.data_width

    h0_dict = hyperopt_sample[0]['h0']
    if h0_dict['h0 layer type'] == 'ConvRectifiedLinear':
        h0, new_data_height, new_data_width = build_conv_rectified_linear(h0_dict, layer_name='h0',
                                                                          data_width=initial_data_width,
                                                                          data_height=initial_data_height)
    else:
        raise ValueError('received layer type:', h0_dict['h0 layer type'], 'is not implemented')

    print t.bold_cyan('\nafter h0:'), '\nheight:', new_data_height, 'width:', new_data_width

    h1_dict = hyperopt_sample[1]['h1']
    if h1_dict is None:
        h1 = None
    elif h1_dict['h1 layer type'] == 'ConvRectifiedLinear':
        h1, new_data_height, new_data_width = build_conv_rectified_linear(h1_dict, layer_name='h1',
                                                                          data_height=new_data_height,
                                                                          data_width=new_data_width)
    else:
        raise ValueError('received layer type:', h1_dict['h1 layer type'], 'is not implemented')

    print t.bold_cyan('\nafter h1:'), '\nheight:', new_data_height, 'width:', new_data_width

    softmax = models.Softmax()
    softmax.n_classes = 2
    softmax.layer_name = 'softmax'
    softmax.irange = 0.05

    if h1 is not None:
        layers = [h0, h1, softmax]
    else:
        layers = [h0, softmax]

    # put layers into MLP

    # TODO: add num channels and batch size to search space maybe?
    space = models.Conv2DSpace()
    space.shape = [initial_data_height, initial_data_width]
    space.num_channels = 1
    space.axes = 'c', 0, 1, 'b'     # there was: ['c', 0, 1, 'b'] I am not sure if this correction is correct

    mlp = models.MLP()
    mlp.batch_size = 1
    mlp.input_space = space
    mlp.layers = layers

    return mlp


def build_conv_rectified_linear(dictionary, layer_name, data_height, data_width):
    crl = models.ConvRectifiedLinear()
    # hp.randint returns numpy array, we need an int here thus casting
    crl.output_channels = 1 + int(dictionary[layer_name + ' output channels'])

    # convolution_matrix shape and stride
    # data_shape -1 as pylearn requires convolution_matrix size smaller than data size
    convolution_matrix_shape_height = min(data_height-1, dictionary[layer_name + ' convolution_matrix shape height'])
    convolution_matrix_shape_width = min(data_width-1, dictionary[layer_name + ' convolution_matrix shape width'])
    crl.kernel_shape = (convolution_matrix_shape_height, convolution_matrix_shape_width)

    # min must be convolution_matrix_shape_height as it might have been reduced because the size of the data
    convolution_matrix_stride_height = min(data_height-convolution_matrix_shape_height,
                                           convolution_matrix_shape_height,
                                           dictionary[layer_name + ' convolution_matrix stride height'])
    convolution_matrix_stride_width = min(data_width-convolution_matrix_shape_width,
                                          convolution_matrix_shape_width,
                                          dictionary[layer_name + ' convolution_matrix stride width'])
    crl.kernel_stride = (convolution_matrix_stride_height, convolution_matrix_stride_width)

    # pooling matrix shape and stride
    pool_shape_width, pool_shape_height = dictionary[layer_name + ' pool shape']
    # data_shape - 1 as pylearn does not allow to have pooling shape equal to data shape
    crl.pool_shape = (min(data_height-1, pool_shape_height), min(data_width-1, pool_shape_width))
    pool_shape_height, pool_shape_width = crl.pool_shape

    pool_stride_height_multiplier = dictionary[layer_name + ' pool stride height']  # this will be 0.5 or 1
    pool_stride_height = max(1, int(pool_shape_height*pool_stride_height_multiplier))
    # nie musi byc normalizowane do data bo i tak jest normalizowane do pool shape

    pool_stride_width_multiplier = dictionary[layer_name + ' pool stride width']  # this will be 0.5 or 1
    pool_stride_width = max(1, int(pool_shape_width*pool_stride_width_multiplier))
    crl.pool_stride = (pool_stride_height, pool_stride_width)

    crl.layer_name = layer_name
    crl.irange = 0.05
    crl.border_mode = 'valid'
    crl.sparse_init = None
    crl.include_prob = 1.0
    crl.init_bias = 0.0
    crl.W_lr_scale = None
    crl.b_lr_scale = None
    crl.left_slope = 0.01
    crl.pool_type = 'max'
    crl.tied_b = False
    crl.detector_normalization = None
    crl.output_normalization = None
    crl.monitor_style = 'classification'

    print t.bold_magenta("layer name: "+str(layer_name)
                         + "\nconvolution_matrix shape: " + str(crl.kernel_shape)
                         + "\npool shape: " + str(crl.pool_shape)
                         + "\t\tconvolution_matrix stride: " + str(crl.kernel_stride)
                         + "\t\tpool_stride: " + str(crl.pool_stride)
                         + "\noutput channels: " + str(crl.output_channels)
                         )

    new_data_height, new_data_width = \
        calculate_new_data_shape(data_height, data_width,
                                 convolution_matrix_shape_height, convolution_matrix_shape_width,
                                 convolution_matrix_stride_height, convolution_matrix_stride_width,
                                 pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width)

    return crl, new_data_height, new_data_width


def calculate_new_data_shape(data_height, data_width,
                             convolution_matrix_shape_height, convolution_matrix_shape_width,
                             convolution_matrix_stride_height, convolution_matrix_stride_width,
                             pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width):
    # convoluting
    new_data_height, new_data_width = \
        calculate_data_shape_after_convolution(data_height, data_width,
                                               convolution_matrix_shape_height,
                                               convolution_matrix_shape_width,
                                               convolution_matrix_stride_height,
                                               convolution_matrix_stride_width)
    # pooling
    new_data_height = 1 + floor((new_data_height-pool_shape_height)/pool_stride_height)
    new_data_width = 1 + floor((new_data_width-pool_shape_width)/pool_stride_width)
    print '\nafter convoluting and pooling', '\nnew_data_height', new_data_height, '\nnew_data_width', new_data_width
    return int(new_data_height), int(new_data_width)


def calculate_data_shape_after_convolution(data_height, data_width,
                                           convolution_matrix_shape_height,
                                           convolution_matrix_shape_width,
                                           convolution_matrix_stride_height,
                                           convolution_matrix_stride_width,):
    new_data_height = 1 + floor((data_height-convolution_matrix_shape_height)/convolution_matrix_stride_height)
    new_data_width = 1 + floor((data_width-convolution_matrix_shape_width)/convolution_matrix_stride_width)
    print '\nafter pooling', '\nnew_data_height', new_data_height, '\nnew_data_width', new_data_width
    return int(new_data_height), int(new_data_width)


# returns object that can be parsed to yaml
def build_multilayer(hyperopt_sample):
    single_sample_size = 20844
    current_data_size = single_sample_size

    h0_dict = hyperopt_sample[0]['h0']
    if h0_dict['h0 layer type'] == 'RectifiedLinear':
        h0 = models.RectifiedLinear()
    elif h0_dict['h0 layer type'] == 'Sigmoid':
        h0 = models.Sigmoid()
    elif h0_dict['h0 layer type'] == 'Tanh':
        h0 = models.Tanh()
    else:
        raise ValueError('received layer type:', h0_dict['h0 layer type'], 'is not implemented')
    h0.dim = int(current_data_size * h0_dict['h0 layer size multiplier'])
    h0.layer_name = 'h0'
    h0.irange = 0.05
    current_data_size = int(current_data_size * h0_dict['h0 layer size multiplier'])

    print t.bold_cyan('\nafter h0:'), '\ndata size:', current_data_size

    h1_dict = hyperopt_sample[1]['h1']
    if h1_dict is None:
        h1 = None
    else:
        if h1_dict['h1 layer type'] == 'RectifiedLinear':
            h1 = models.RectifiedLinear()
        elif h1_dict['h1 layer type'] == 'Sigmoid':
            h1 = models.Sigmoid()
        elif h1_dict['h1 layer type'] == 'Tanh':
            h1 = models.Tanh()
        else:
            raise ValueError('received layer type:', h1_dict['h1 layer type'], 'is not implemented')
        h1.dim = int(current_data_size * h1_dict['h1 layer size multiplier'])
        h1.layer_name = 'h1'
        h1.irange = 0.05
        current_data_size = int(current_data_size * h1_dict['h1 layer size multiplier'])

    print t.bold_cyan('\nafter h1:'), '\ndata:', current_data_size

    h2_dict = hyperopt_sample[2]['h2']
    if h2_dict is None:
        h2 = None
    else:
        if h2_dict['h2 layer type'] == 'RectifiedLinear':
            h2 = models.RectifiedLinear()
        elif h2_dict['h2 layer type'] == 'Sigmoid':
            h2 = models.Sigmoid()
        elif h2_dict['h2 layer type'] == 'Tanh':
            h2 = models.Tanh()
        else:
            raise ValueError('received layer type:', h2_dict['h2 layer type'], 'is not implemented')
        h2.dim = int(current_data_size * h2_dict['h2 layer size multiplier'])
        h2.layer_name = 'h2'
        h2.irange = 0.05
        current_data_size = int(current_data_size * h2_dict['h2 layer size multiplier'])

    print t.bold_cyan('\nafter h2:'), '\ndata:', current_data_size

    softmax = models.Softmax()
    softmax.n_classes = 2
    softmax.layer_name = 'softmax'
    softmax.irange = 0.05

    layers = [h0]
    if h1 is not None:
        layers.append(h1)
    if h2 is not None:
        layers.append(h2)
    layers.append(softmax)

    # put layers into MLP
    mlp = models.MLP()
    mlp.batch_size = 1
    mlp.nvis = single_sample_size
    mlp.layers = layers

    return mlp

