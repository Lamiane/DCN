import yaml_parser as yp
from pylearn2.config import yaml_parse

with open("example.yaml") as f:
    yaml_string = f.read()

con = yp.ConvElemwise()
con.layer_name = "con_layer"
con.output_channels = [1, 1]
con.kernel_shape = [2, 2]
con.nonlinearity = yp.TanhConvNonlinearity()
con.irange = 0.1
con.pool_shape = [2, 3]

sof = yp.Softmax()
sof.n_classes = 2
sof.layer_name = "softmax_layer"

layers = [con, sof]

mlp = yp.MLP()
mlp.layers = layers

hyper_params = {'model': yp.parse_to_yaml(mlp)}

yaml_string = yaml_string % hyper_params
# print type(yaml_string)
for number, line in enumerate(yaml_string.split('\n')):
    print line

model = yaml_parse.load(yaml_string)