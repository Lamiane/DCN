__author__ = 'agnieszka'

from math import floor
from blessings import Terminal
import sys
sys.path.append('..')
from yaml_maker import yaml_parser as yp

t = Terminal()


# returns object that can be parsed to yaml
def build(hyperopt_sample):

    initial_data_height = 18    # TODO: it cannot be so!
    initial_data_width = 3492   # 3474 + 2*9


    h0_dict = hyperopt_sample[0]['h0']
    if h0_dict['layer type'] == 'ConvRectifiedLinear':
        h0, new_data_height, new_data_width = build_conv_rectified_linear(h0_dict, layer_name='h0',
                                                                          data_width=initial_data_width,
                                                                          data_height=initial_data_height)
    else:
        raise ValueError('received layer type:', h0_dict['layer type'], 'is not implemented')

    print t.bold_cyan('\nafter h0:'), '\nheight:', new_data_height, 'width:', new_data_width

    h1_dict = hyperopt_sample[1]['h1']
    if h1_dict is None:
        h1 = None
    elif h1_dict['layer type'] == 'ConvRectifiedLinear':
        h1, new_data_height, new_data_width = build_conv_rectified_linear(h1_dict, layer_name='h1',
                                                                          data_height=new_data_height,
                                                                          data_width=new_data_width)
    else:
        raise ValueError('received layer type:', h1_dict['layer type'], 'is not implemented')

    print t.bold_cyan('\nafter h1:'), '\nheight:', new_data_height, 'width:', new_data_width


    softmax = yp.Softmax()
    softmax.n_classes = 2
    softmax.layer_name = 'softmax'
    softmax.irange = 0.05

    if h1 is not None:
        layers = [h0, h1, softmax]
    else:
        layers = [h0, softmax]

    # put layers into MLP

    # TODO: add num channels and batch size to search space maybe?
    space = yp.Conv2DSpace()
    space.shape = [initial_data_height, initial_data_width]
    space.num_channels = 1
    space.axes = 'c', 0, 1, 'b'     # there was: ['c', 0, 1, 'b'] I am not sure if this correction is correct

    mlp = yp.MLP()
    mlp.batch_size = 1
    mlp.input_space = space
    mlp.layers = layers

    return mlp


def build_conv_rectified_linear(dictionary, layer_name, data_height, data_width):
    CRL = yp.ConvRectifiedLinear()
    CRL.output_channels = 1 + int(dictionary[layer_name + ' output channels'])    # hp.randint returns numpy array, we need an int here

    # convolution matrix shape and stride
    # data_shape -1 as pylearn requires convolution matrix size smaller than data size
    kernel_shape_height = min(data_height-1, dictionary[layer_name + ' kernel shape height'])
    kernel_shape_width = min(data_width-1, dictionary[layer_name + ' kernel shape width'])
    CRL.kernel_shape = (kernel_shape_height, kernel_shape_width)

    # min must be kernel_shape_height as it might have been reduced because the size of the data
    kernel_stride_height = min(data_height-kernel_shape_height, kernel_shape_height, dictionary[layer_name + ' kernel stride height'])
    kernel_stride_width = min(data_width-kernel_shape_width, kernel_shape_width, dictionary[layer_name + ' kernel stride width'])
    CRL.kernel_stride = (kernel_stride_height, kernel_stride_width)

    # pooling matrix shape and stride
    pool_shape_width, pool_shape_height = dictionary[layer_name + ' pool shape']
    # data_shape - 1 as pylearn does not allow to have pooling shape equal to data shape
    CRL.pool_shape = (min(data_height-1, pool_shape_height), min(data_width-1, pool_shape_width))
    pool_shape_height, pool_shape_width = CRL.pool_shape

    pool_stride_height_multiplier = dictionary[layer_name + ' pool stride height']  # this will be 0.5 or 1
    pool_stride_height = max(1, int(pool_shape_height*pool_stride_height_multiplier))
    # nie musi byc normalizowane do data bo i tak jest normalizowane do pool shape

    pool_stride_width_multiplier = dictionary[layer_name + ' pool stride width']  # this will be 0.5 or 1
    pool_stride_width = max(1, int(pool_shape_width*pool_stride_width_multiplier))
    CRL.pool_stride = (pool_stride_height, pool_stride_width)

    CRL.layer_name = layer_name
    CRL.irange = 0.05
    CRL.border_mode = 'valid'
    CRL.sparse_init = None
    CRL.include_prob = 1.0
    CRL.init_bias = 0.0
    CRL.W_lr_scale = None
    CRL.b_lr_scale = None
    CRL.left_slope = 0.01
    CRL.pool_type = 'max'
    CRL.tied_b = False
    CRL.detector_normalization = None
    CRL.output_normalization = None
    CRL.monitor_style = 'classification'

    print t.bold_magenta("layer name: "+str(layer_name)
                         + "\nkernel shape: " + str(CRL.kernel_shape) + "\t\tkernel stride: " + str(CRL.kernel_stride)
                         + "\npool shape: " + str(CRL.pool_shape) + "\t\tpool_stride: " + str(CRL.pool_stride))

    new_data_height, new_data_width = \
        calculate_new_data_shape(data_height, data_width,
                                 kernel_shape_height, kernel_shape_width, kernel_stride_height, kernel_stride_width,
                                 pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width)

    return CRL, new_data_height, new_data_width


def calculate_new_data_shape(data_height, data_width,
                             kernel_shape_height, kernel_shape_width, kernel_stride_height, kernel_stride_width,
                             pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width):
    # kerneling
    new_data_height, new_data_width = \
        calculate_data_shape_after_convolution(data_height, data_width, kernel_shape_height, kernel_shape_width,
                                               kernel_stride_height, kernel_stride_width)
    # pooling
    new_data_height = 1 + floor((new_data_height-pool_shape_height)/pool_stride_height)
    new_data_width = 1 + floor((new_data_width-pool_shape_width)/pool_stride_width)
    print '\nafter convoluting and pooling', '\nnew_data_height', new_data_height, '\nnew_data_width', new_data_width
    return int(new_data_height), int(new_data_width)


def calculate_data_shape_after_convolution(data_height, data_width, kernel_shape_height, kernel_shape_width,
                                           kernel_stride_height, kernel_stride_width,):
    new_data_height = 1 + floor((data_height-kernel_shape_height)/kernel_stride_height)
    new_data_width = 1 + floor((data_width-kernel_shape_width)/kernel_stride_width)
    print '\nafter pooling', '\nnew_data_height', new_data_height, '\nnew_data_width', new_data_width
    return int(new_data_height), int(new_data_width)


