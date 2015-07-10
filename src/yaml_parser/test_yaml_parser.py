__author__ = 'agnieszka'
import models
from yaml_parser import parse_to_yaml


def t1():
    l = [models.MLP(), models.Linear(), models.TanhConvNonlinearity(), models.SoftmaxPool()]
    return parse_to_yaml(l)


def t2():
    d = {'a': models.MLP(), 'b': models.Linear(), 'c': models.TanhConvNonlinearity(), 'd': models.SoftmaxPool()}
    return parse_to_yaml(d)


def t3():
    tup = (models.MLP(), models.Linear(), models.TanhConvNonlinearity(), models.SoftmaxPool())
    return parse_to_yaml(tup)


def t():
    print "\n\nrunning t1"
    print t1()
    print "\n\nrunning t2"
    print t2()
    print "\n\nrunning t3"
    print t3()
