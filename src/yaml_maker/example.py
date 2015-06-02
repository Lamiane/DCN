import yaml_parser as yp
from pylearn2.config import yaml_parse
import sys
sys.path.append('..')
from tmp import notify

# setting parameters of convolutional layer
con = yp.ConvElemwise()
con.layer_name = "con_layer"
con.output_channels = 1
con.kernel_shape = [2, 2]
con.nonlinearity = yp.TanhConvNonlinearity()
con.irange = 0.1
con.pool_shape = [2, 3]

# setting parameters of softmax layer
sof = yp.Softmax()
sof.n_classes = 10
sof.layer_name = "softmax_layer"
sof.irange = 0.1

# creating list of layers
layers = [con, sof]

# creating space
some_space = yp.Conv2DSpace()
some_space.axes = ['c', 0, 1, 'b']
some_space.shape = [28, 28]
some_space.num_channels = 1

# setting parameters of MLP model
mlp = yp.MLP()
mlp.layers = layers
mlp.input_space = some_space

# creating a dictionary with hyperparameters
hyper_params = {'model': yp.parse_to_yaml(mlp)}

# obtaining the yaml skelton
with open("example.yaml") as f:
    default_string = f.read()

# filling the yaml skelton with hyperparameters
yaml_string = default_string % hyper_params


with open('generated_yaml.yaml', 'w') as g:
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

