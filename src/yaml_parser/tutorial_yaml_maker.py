__author__ = 'agnieszka'

from os.path import join
from pylearn2.config import yaml_parse
import sys
sys.path.append('..')
from utils.common import notify
import configuration.model as config
from utils.common import get_timestamp
import models
import yaml_parser as yp
from utils.common import get_timestamp

current_time = get_timestamp()

# setting parameters of convolutional layer
con = models.ConvElemwise()
con.layer_name = "h0"
con.output_channels = 1
con.kernel_shape = [2, 2]
con.nonlinearity = models.TanhConvNonlinearity()
con.irange = 0.1
con.pool_shape = [2, 3]

# setting parameters of softmax layer
sof = models.Softmax()
sof.n_classes = 10
sof.layer_name = "softmax"
sof.irange = 0.1

# creating list of layers
layers = [con, sof]

# creating space
some_space = models.Conv2DSpace()
some_space.axes = ['c', 0, 1, 'b']
some_space.shape = [28, 28]
some_space.num_channels = 1

# setting parameters of MLP model
mlp = models.MLP()
mlp.layers = layers
mlp.input_space = some_space

# define weight decay parameters. They depend on the number of layers (there is one parameter fo each layer)
weight_decay_coeffs = yp.parse_weight_decay(mlp)

# define data paths
path = config.data_path

# generate a filename to store the best model
pkl_filename = join(config.path_for_storing, current_time+"example_best.pkl")

# create dictionary with hyper parameters
hyper_params = {'model': yp.parse_to_yaml(mlp), 'path': yp.parse_to_yaml(path),
                'weight_decay_coeffs': weight_decay_coeffs, 'pkl_filename': pkl_filename}

# obtaining the yaml skelton
with open(config.yaml_skelton_path) as f:
    default_string = f.read()

# filling the yaml skelton with hyperparameters
yaml_string = default_string % hyper_params

generated_yaml_path = join(config.path_for_storing, current_time+'generated_yaml.yaml')
with open(generated_yaml_path, 'w') as g:
    g.write(yaml_string)

# creating the model based on a yaml
model = yaml_parse.load(yaml_string)

# training the model
model.main_loop()

# the settings can be changed in a loop
# for i in range(0, 10):
#     mlp.layers[1].n_classes = i
#     hyper_params = {'model':yp.parse_to_yaml(mlp)}

notify()

