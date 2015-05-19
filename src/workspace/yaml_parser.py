__author__ = 'agnieszka'


class YamlParser(object):

    def __init__(self):
        self.yaml_path = None

        # dataset
        self.dataset_path = None # or not a path

        # model
        self.mlp_batch_size = None
        self.input_space = None
        self.mlp_input_space_shape = None           # to jest w kazdym input spejsie? ogarnac dokumentacje!!!
        self.mlp_input_space_num_channels = None    # to jest w kazdym input spejsie? ogarnac dokumentacje!!!

        # models layers
        self.layers = [] # an ordered list


        # algorithm


        # extensions
        self.dropout = False    # default: false


class Linear(object):

    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.Linear"

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
        self.hierarchy = "pylearn2.models.mlp.LinearGaussian"

        # these must be completed
        self.init_beta = None
        self.min_beta = None
        self.max_beta = None
        self.beta_lr_scale = None
        self.kwargs = None


class RectifiedLinear(object):
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.RectifiedLinear"

        # these have default values
        self.left_slope = 0.0
        self.kwargs = None


class ConvElemwise(object):

    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.ConvElemwise"

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


class Softmax(object):

    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.Softmax"

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
        self.hierarchy = "pylearn2.models.mlp.ConvRectifiedLinear"

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
        self.hierarchy = "pylearn2.models.mlp.SoftmaxPool"

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
        self.hierarchy = "pylearn2.models.mlp.Sigmoid"

        # these have default values
        self.monitor_style = 'detection'
        self.kwargs = None


class Tanh(object):
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.Tanh"

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
        self.hierarchy = "pylearn2.models.mlp.RectifierConvNonlinearity"
        # these have default values
        self.left_slope = 0.0


class SigmoidConvNonlinearity():
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.SigmoidConvNonlinearity"
        # these have default values
        self.monitor_style = 'classification'


# nieparametryzowalne obiekty:
class ConvNonlinearity(object):
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.ConvNonlinearity"


class IdentityConvNonlinearity(object):
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.IdentityConvNonlinearity"


class TanhConvNonlinearity(object):
    def __init__(self):
        self.hierarchy = "pylearn2.models.mlp.TanhConvNonlinearity"
