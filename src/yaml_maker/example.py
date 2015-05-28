import yaml_parser as yp
from pylearn2.config import yaml_parse

<<<<<<< HEAD
# setting parameters of convolutional layer
=======
with open("example.yaml") as f:
    yaml_string = f.read()

>>>>>>> 060981ca14b04c87e03eed3d62e4a69eeec724de
con = yp.ConvElemwise()
con.layer_name = "con_layer"
con.output_channels = [1, 1]
con.kernel_shape = [2, 2]
con.nonlinearity = yp.TanhConvNonlinearity()
con.irange = 0.1
con.pool_shape = [2, 3]

<<<<<<< HEAD
# setting parameters of softmax layer
=======
>>>>>>> 060981ca14b04c87e03eed3d62e4a69eeec724de
sof = yp.Softmax()
sof.n_classes = 2
sof.layer_name = "softmax_layer"

<<<<<<< HEAD
# creating list of layers
layers = [con, sof]

# setting parameters of MLP model
mlp = yp.MLP()
mlp.layers = layers

# creating a dictionary with hyperparameters
hyper_params = {'model': yp.parse_to_yaml(mlp)}

# obtaining the yaml skelton
with open("example.yaml") as f:
    pass

default_string = f.read()

# filling the yaml skelton with hyperparameters
yaml_string = default_string % hyper_params


# print type(yaml_string)
# for number, line in enumerate(yaml_string.split('\n')):
#     print line

# creating the model based on a yaml
model = yaml_parse.load(yaml_string)

# training the model
model.main_loop()

# the setting can be changed in a loop :)
# for i in range(0, 10):
#     mlp.layers[1].n_classes = i
#     hyper_params = {'model':yp.parse_to_yaml(mlp)}
=======
layers = [con, sof]

mlp = yp.MLP()
mlp.layers = layers

hyper_params = {'model': yp.parse_to_yaml(mlp)}

yaml_string = yaml_string % hyper_params
# print type(yaml_string)
for number, line in enumerate(yaml_string.split('\n')):
    print line

model = yaml_parse.load(yaml_string)
>>>>>>> 060981ca14b04c87e03eed3d62e4a69eeec724de
