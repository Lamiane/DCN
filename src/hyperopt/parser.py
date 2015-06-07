from math import floor
from blessings import Terminal
import sys
sys.path.append('..')
from yaml_maker import yaml_parser as yp

t = Terminal()

# returns object that can be parsed to yaml
def build(hyperopt_sample):
    initial_data_height = 3474   # TODO: it cannot be so!
    initial_data_width = 6

    h0_dict = hyperopt_sample[0]['h0']
    if h0_dict['layer type'] == 'ConvRectifiedLinear':
        h0, new_data_height, new_data_width = build_conv_rectified_linear(h0_dict, layer_name='h0',
                                                                          data_width=initial_data_width,
                                                                          data_height=initial_data_height)
    elif h0_dict['layer type'] == 'ConvElementWise':
        h0, new_data_height, new_data_width = build_conv_elementwise(h0_dict, layer_name='h0',
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
    elif h1_dict['layer type'] == 'ConvElementWise':
        h1, new_data_height, new_data_width = build_conv_elementwise(h1_dict, layer_name='h1',
                                                                     data_height=new_data_height,
                                                                     data_width=new_data_width)
    else:
        raise ValueError('received layer type:', h1_dict['layer type'], 'is not implemented')

    output_layer = build_softmax(hyperopt_sample[2], 'softmax')

    if h1 is not None:
        layers = [h0, h1, output_layer]
    else:
        layers = [h0, output_layer]

    # put layers into MLP

    # TODO: implement fajanse

    # TODO: add num channels and batch size to search space maybe?
    space = yp.Conv2DSpace()
    space.shape = [6, 3474]
    space.num_channels = 1
    space.axes = ['c', 0, 1, 'b']

    mlp = yp.MLP()
    mlp.batch_size = 1
    mlp.input_space = space
    mlp.layers = layers

    return mlp


def build_conv_rectified_linear(dictionary, layer_name, data_height, data_width):
    CRL = yp.ConvRectifiedLinear()
    CRL.output_channels = 1 + int(dictionary['output channels'])    # hp.randint returns numpy array, we need an int here

    kernel_shape_height = max(1, round(dictionary['kernel shape height']*data_height))
    kernel_shape_width = max(1, round(dictionary['kernel shape width']*data_width))
    CRL.kernel_shape = (int(kernel_shape_width), int(kernel_shape_height))

    kernel_stride_height = max(1, round(dictionary['kernel stride height'] * kernel_shape_height))
    kernel_stride_width = max(1, round(dictionary['kernel stride width'] * kernel_shape_width))
    CRL.kernel_stride = (int(kernel_stride_width), int(kernel_stride_height))

    data_height_after_convolution, data_width_after_convolution =\
        calculate_data_shape_after_convolution(data_height, data_width, kernel_shape_height, kernel_shape_width,
                                               kernel_stride_height, kernel_stride_width)
    pool_shape_height = max(1, round(dictionary['pool shape height'] * data_height_after_convolution))
    pool_shape_width = max(1, round(dictionary['pool shape width'] * data_width_after_convolution))
    CRL.pool_shape = (int(pool_shape_width), int(pool_shape_height))

    pool_stride_height = max(1, round(dictionary['pool stride height'] * pool_shape_height))
    pool_stride_width = max(1, round(dictionary['pool stride width']) * pool_shape_width)
    CRL.pool_stride = (int(pool_stride_width), int(pool_stride_height))

    CRL.layer_name = layer_name
    CRL.irange = dictionary['irange']
    CRL.border_mode = 'valid'   # TODO: it's OK, isn' it?
    CRL.sparse_init = None
    CRL.include_prob = 1.0
    CRL.init_bias = 0.0
    CRL.W_lr_scale = None
    CRL.b_lr_scale = None
    CRL.left_slope = 0.0
    CRL.max_kernel_norm = dictionary['max kernel norm']
    CRL.pool_type = 'max'
    CRL.tied_b = False
    CRL.detector_normalization = None
    CRL.output_normalization = None
    CRL.monitor_style = 'classification'
    new_data_height, new_data_width = \
        calculate_new_data_shape(data_height, data_width,
                                 kernel_shape_height, kernel_shape_width, kernel_stride_height, kernel_stride_width,
                                 pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width)
    return CRL, new_data_height, new_data_width


def build_conv_elementwise(dictionary, layer_name, data_height, data_width):
    CEW = yp.ConvElemwise()
    CEW.output_channels = 1 + int(dictionary['output channels'])    # hp.randint returns numpy array, we need an int here

    kernel_shape_height = max(1, round(dictionary['kernel shape height']*data_height))
    kernel_shape_width = max(1, round(dictionary['kernel shape width']*data_width))
    CEW.kernel_shape = (int(kernel_shape_width), int(kernel_shape_height))

    kernel_stride_height = max(1, round(dictionary['kernel stride height'] * kernel_shape_height))
    kernel_stride_width = max(1, round(dictionary['kernel stride width'] * kernel_shape_width))
    CEW.kernel_stride = (int(kernel_stride_width), int(kernel_stride_height))

    data_height_after_convolution, data_width_after_convolution =\
        calculate_data_shape_after_convolution(data_height, data_width, kernel_shape_height, kernel_shape_width,
                                               kernel_stride_height, kernel_stride_width)
    pool_shape_height = max(1, round(dictionary['pool shape height'] * data_height_after_convolution))
    pool_shape_width = max(1, round(dictionary['pool shape width'] * data_width_after_convolution))
    CEW.pool_shape = (int(pool_shape_width), int(pool_shape_height))

    pool_stride_height = max(1, round(dictionary['pool stride height'] * pool_shape_height))
    pool_stride_width = max(1, round(dictionary['pool stride width']) * pool_shape_width)
    CEW.pool_stride = (int(pool_stride_width), int(pool_stride_height))

    CEW.layer_name = layer_name
    CEW.nonlinearity = build_nonlinearity(dictionary['nonlinearity'])
    CEW.pool_type = "max"
    CEW.irange = dictionary['irange']
    CEW.max_kernel_norm = dictionary['max kernel norm']
    CEW.border_mode = 'valid'
    CEW.sparse_init = None
    CEW.include_prob = 1.0
    CEW.init_bias = 0.0
    CEW.W_lr_scale = None
    CEW.b_lr_scale = None
    CEW.tied_b = None
    CEW.detector_normalization = None
    CEW.output_normalization = None
    CEW.monitor_style = 'classification'
    new_data_height, new_data_width = \
        calculate_new_data_shape(data_height, data_width,
                                 kernel_shape_height, kernel_shape_width, kernel_stride_height, kernel_stride_width,
                                 pool_shape_height, pool_shape_width, pool_stride_height, pool_stride_width)
    return CEW, new_data_height, new_data_width


def build_nonlinearity(dictionary):
    if dictionary['nonlinearity type'] == 'TanhConvNonlinearity':
        nonlinearity = yp.TanhConvNonlinearity()
    elif dictionary['nonlinearity type'] == 'SigmoidConvNonlinearity':
        nonlinearity = yp.SigmoidConvNonlinearity()
    elif dictionary['nonlinearity type'] == 'RectifierConvNonlinearity':
        nonlinearity = yp.RectifierConvNonlinearity()
        nonlinearity.left_slope = dictionary['left slope']
    else:
        raise ValueError('received nonlinearity type:', dictionary['nonlinearity type'], 'is not implemented')

    return nonlinearity


def build_softmax(dictionary, layer_name):
    softmax = yp.Softmax()
    softmax.n_classes = 3   # set as 3 because we won't be changing this
    softmax.layer_name = layer_name
    softmax.irange = dictionary['irange']
    return softmax


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


