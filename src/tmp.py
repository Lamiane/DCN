import subprocess as s
from pylearn2.config import yaml_parse


# will stop playing after entering 'q' and hitting Enter
def notify():
    s.call(["mplayer", "-slave", "-quiet", "../../media/crazy_mexican.mp3"])


def run_yaml(path):
    with open(path) as yaml_file:
        yaml = yaml_file.read()

    network = yaml_parse.load(yaml)
    network.main_loop()
    notify()