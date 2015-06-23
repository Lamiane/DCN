__author__ = 'agnieszka'
from re import compile, match


def is_valid_attribute(name):
    pattern = compile("[A-Za-z]") # nie bierzemy zadnych metod ani pol ktore zaczynaja sie od __
    if pattern.match(name) is None:
        return False
    return True


def parse_to_yaml(obj, tabulators=0):
    tabs = 4*' '
    output = tabulators*tabs
    if type(obj) is type(None):
        return 'None' # this shall never happen, how to deal with it? raise an exception?
    elif type(obj) in (type([]), type(())):
        output += "["
        for element in obj:
            output += parse_to_yaml(element, tabulators+1)
        output += "],"
    elif type(obj) is type({}):
        output += "{"
        for keys, vals in obj.iteritems():
            output += (tabulators+1)*tabs + keys + ": " + parse_to_yaml(vals, 0)
        output += tabulators*tabs + "},\n"
    elif isinstance(obj, str):
        output = obj.__str__() + ","
    elif type(obj) in [type(0), type(0.0), type(True)]:
        output = obj.__repr__() + ", "
    else:
        list_of_parameters = filter(is_valid_attribute, dir(obj))
        output = "!obj:" + obj.__hierarchy__ + " {"
        for element in list_of_parameters:
            val = getattr(obj, element)
            if val is not None:
                output += "\n" + (tabulators+1)*tabs + element + ": " + parse_to_yaml(val, tabulators+1)
        output += '\n' + tabulators*tabs + '},\n'

    return output



class YamlParser(object):

    def __init__(self):
        self.yaml_path = None

        # dataset
        self.dataset_path = None    # or not a path

        # model
        self.model = MLP()

        # models layers
        self.layers = []    # an ordered list

        # algorithm

        # extensions
        self.dropout = False    # default: false


# Models
class MLP(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.MLP"

        # these must be completed
        self.layers = None

        # these have default values
        self.batch_size = None
        self.input_space = None
        self.input_source = 'features'
        self.target_source = 'targets'
        self.nvis = None
        self.seed = None
        self.layer_name = None
        self.monitor_targets = True
        self.kwargs = None

    def __str__(self):
        return "layers:" + str(self.layers) + \
               "\nbatch size:" + str(self.batch_size) + \
               "\ninput space:" + str(self.input_space)

    def __repr__(self):
        return self.__str__()


# Layers
class Linear(object):

    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.Linear"

        # these must be completed
        self.dim = None
        self.layer_name = None

        # these have default values
        self.irange = None
        self.istdev = None
        self.sparse_init = None
        self.sparse_stdev = 1.0
        self.include_prob = 1.0
        self.init_bias = 0.0
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.mask_weights = None
        self.max_row_norm = None
        self.max_col_norm = None
        self.min_col_norm = None
        self.copy_input = None
        self.use_abs_loss = False
        self.use_bias = True


class LinearGaussian:
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.LinearGaussian"

        # these must be completed
        self.init_beta = None
        self.min_beta = None
        self.max_beta = None
        self.beta_lr_scale = None
        self.kwargs = None


class RectifiedLinear(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.RectifiedLinear"

        # these have default values
        self.left_slope = 0.0
        self.kwargs = None


class ConvElemwise(object):

    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.ConvElemwise"

        # these must be completed
        self.output_channels = None
        self.kernel_shape = None
        self.layer_name = None
        self.nonlinearity = None

        # these have default values
        self.pool_type = "max"  # default: max pooling, different than in pylearn
        self.pool_shape = None
        self.pool_stride = (1, 1)
        self.irange = None
        self.max_kernel_norm = None
        self.border_mode = 'valid'
        self.sparse_init = None
        self.include_prob = 1.0
        self.init_bias = 0.0
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.tied_b = None
        self.detector_normalization = None
        self.output_normalization = None
        self.monitor_style = 'classification'
        self.kernel_stride = (1, 1)


class Softmax(object):

    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.Softmax"

        # these must be completed
        self.n_classes = None
        self.layer_name = None

        # these have default values
        self.irange = None
        self.istdev = None
        self.sparse_init = None
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.max_row_norm = None
        self.no_affine = False
        self.max_col_norm = None
        self.init_bias_target_marginals = None
        self.binary_target_dim = None
        self.non_redundant = False


class ConvRectifiedLinear(object):

    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.ConvRectifiedLinear"

        # these must be completed
        self.output_channels = None
        self.kernel_shape = None
        self.pool_shape = None
        self.pool_stride = None
        self.layer_name = None

        # these have default values
        self.irange = None
        self.border_mode = 'valid'
        self.sparse_init = None
        self.include_prob = 1.0
        self.init_bias = 0.0
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.left_slope = 0.0
        self.max_kernel_norm = None
        self.pool_type = 'max'
        self.tied_b = False
        self.detector_normalization = None
        self.output_normalization = None
        self.kernel_stride = (1, 1)
        self.monitor_style = 'classification'


class SoftmaxPool(object):

    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.SoftmaxPool"

        # these must be completed
        self.detector_layer_dim = None
        self.layer_name = None

        # these have default values
        self.pool_size = 1
        self.irange = None
        self.sparse_init = None
        self.sparse_stdev = 1.0
        self.include_prob = 1.0
        self.init_bias = 0.0
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.mask_weights = None
        self.max_col_norm = None


class Sigmoid(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.Sigmoid"

        # these have default values
        self.monitor_style = 'detection'
        self.kwargs = None


class Tanh(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.Tanh"

        # these must be completed
        self.dim = None
        self.layer_name = None

        # these have default values
        self.irange = None
        self.istdev = None
        self.sparse_init = None
        self.sparse_stdev = 1.0
        self.include_prob = 1.0
        self.init_bias = 0.0
        self.W_lr_scale = None
        self.b_lr_scale = None
        self.mask_weights = None
        self.max_row_norm = None
        self.max_col_norm = None
        self.min_col_norm = None
        self.copy_input = None
        self.use_abs_loss = False
        self.use_bias = True


class RectifierConvNonlinearity(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.RectifierConvNonlinearity"
        # these have default values
        self.left_slope = 0.0


class SigmoidConvNonlinearity():
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.SigmoidConvNonlinearity"
        # these have default values
        self.monitor_style = 'classification'


# nieparametryzowalne obiekty:
class ConvNonlinearity(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.ConvNonlinearity"


class IdentityConvNonlinearity(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.IdentityConvNonlinearity"


class TanhConvNonlinearity(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.models.mlp.TanhConvNonlinearity"

# Spaces


class CompositeSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.CompositeSpace"

        # these must be completed
        self.components = None
        # these have default values
        self.kwargs = None


class Conv2DSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.Conv2DSpace"

        # these must be completed
        self.shape = None

        # these have default values
        self.channels = None
        self.num_channels = None
        self.axes = None
        self.dtype = 'floatX'
        self.kwargs = None


class IndexSequenceSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.IndexSequenceSpace"

        # these must be completed
        self.max_labels = None
        self.dim = None
        # these have default values
        self.dtype = 'int64'
        self.kwargs = None


class IndexSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.IndexSpace"

        # these must be completed
        self.max_labels = None
        self.dim = None
        # these have default values
        self.dtype = 'int64'
        self.kwargs = None


class SimplyTypedSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.SimplyTypedSpace"

        # these must be completed
        # these have default values
        self.dtype = 'floatX'
        self.kwargs = None


class VectorSequenceSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.VectorSequenceSpace"

        # these must be completed
        # these have default values
        self.dtype = 'floatX'
        self.kwargs = None


class VectorSpace(object):
    def __init__(self):
        self.__hierarchy__ = "pylearn2.space.VectorSpace"

        # these must be completed
        self.dim = None
        # these have default values
        self.sparse = False
        self.dtype = 'floatX'
        self.kwargs = None


# class (object):
#     def __init__(self):
#         self.__hierarchy__ = ""
#
#         # these must be completed
#
#         # these have default values
#
# class (object):
#     def __init__(self):
#         self.__hierarchy__ = ""
#
#         # these must be completed
#
#         # these have default values
#
# class (object):
#     def __init__(self):
#         self.__hierarchy__ = ""
#
#         # these must be completed
#
#         # these have default values
#
# class (object):
#     def __init__(self):
#         self.__hierarchy__ = ""
#
#         # these must be completed
#
#         # these have default values
