import yaml_parser as Y


def t1():
    l = [Y.MLP(), Y.Linear(), Y.TanhConvNonlinearity(), Y.SoftmaxPool()]
    return Y.parse_to_yaml(l)


def t2():
    d = {'a': Y.MLP(), 'b': Y.Linear(), 'c': Y.TanhConvNonlinearity(), 'd': Y.SoftmaxPool()}
    return Y.parse_to_yaml(d)


def t3():
    tup = (Y.MLP(), Y.Linear(), Y.TanhConvNonlinearity(), Y.SoftmaxPool())
    return Y.parse_to_yaml(tup)


def t():
    print "\n\nrunning t1"
    print t1()
    print "\n\nrunning t2"
    print t2()
    print "\n\nrunning t3"
    print t3()
