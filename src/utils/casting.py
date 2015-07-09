__author__ = 'agnieszka'


def pkl2model(filepath):
    from pylearn2.utils import serial
    model = serial.load(filepath)
    return model


def yaml2train(yaml_path):
    import traceback
    from pylearn2.config import yaml_parse

    # obtain the yaml skelton
    with open(yaml_path) as f:
        yaml_string = f.read()

    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
    except BaseException as e:
        print traceback.format_exc()
    return network
