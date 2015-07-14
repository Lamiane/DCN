__author__ = 'agnieszka'


def pkl2model(filepath):
    # TODO: test if needed. Perhaps calling this method after running python shell from utils directory will fail
    import sys
    sys.path.append('..')
    from pylearn2.utils import serial
    model = serial.load(filepath)
    return model


def yaml2train(yaml_path):
    import traceback
    from pylearn2.config import yaml_parse
    # TODO: test if needed. Perhaps calling this method after running python shell from utils directory will fail
    import sys
    sys.path.append('..')

    # obtain the yaml skelton
    with open(yaml_path) as f:
        yaml_string = f.read()

    try:
        # create the model based on a yaml
        network = yaml_parse.load(yaml_string)
        return network
    except BaseException as e:
        print traceback.format_exc()
    print "Returning None."
    return None
