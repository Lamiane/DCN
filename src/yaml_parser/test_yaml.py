__author__ = 'agnieszka'


# if it's running then it work, otherwise exceptions will be thrown
def run_yaml(path):
    import sys
    sys.path.append('..')
    from pylearn2.config import yaml_parse
    from utils.common import notify

    with open(path) as yaml_file:
        yaml = yaml_file.read()

    network = yaml_parse.load(yaml)
    network.main_loop()
    notify()
