__author__ = 'agnieszka'


# only for one threshold
def types_dict(y_true, y_predicted, threshold=0.5):
    import sys
    sys.path.append('..')
    from utils import values

    # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
    # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
    # middle    -1    [[ 0. 0. 1. ]]

    axis = zip(y_true, y_predicted)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    FNN = 0
    FNP = 0

    for tr, pre in axis:
        if tr[0] == 1:
            if pre[0][1] >= threshold:
                TP += 1
            else:
                FN += 1
        else:
            if pre[0][1] < threshold:
                TN += 1
            else:
                FP += 1

    return {values.TP: TP, values.FP: FP, values.TN: TN, values.FN: FN, values.FNN: FNN, values.FNP: FNP}


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
    except BaseException:
        print traceback.format_exc()
    print "Returning None."
    return None


def label_lists2types(y_true, y_predicted, sym_t=None, t0=None, t1=None):
    import sys
    sys.path.append('..')
    from utils import values

    # checking if parameters are valid
    if (t0 is None and t1 is not None) or (t1 is None and t0 is not None):
        raise ValueError("t1 and t2 must be both provided or neither of them can be provided")
    if sym_t is not None and t0 is not None:
        raise ValueError("If any threshold need to be provided then either sym_t must be provided either t1 and t2")

    # t0 < t1

    # setting t0 and t1 according to provided values
    if sym_t is None and t0 is None and t1 is None:
        # no threshold has been provided, initialising default value
        t0 = 0.5
        t1 = 0.5
    elif sym_t is not None:
        if not 0 < sym_t < 1:
            raise ValueError("sym_t must be in (0, 1)")
        t0 = sym_t
        t1 = sym_t
    else:
        # both t1 and t2 has been provided
        if not (0 < t0 < 1 and 0 < t1 < 1):
            raise ValueError("t0 and t1 must be in (0, 1)")
        # t0 = t0
        # t1 = t1

    # [0, t] - not classified
    # (t, 1] - positive
    # computation
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    FNN = 0
    FNP = 0
    for index in xrange(len(y_true)):
        y_t = y_true[index]             # this is a label (a scalar value, 0 or 1)
        y_p = y_predicted[index][0]     # these are predictions - a vector of two scalar values in range [0, 1]

        if y_t == 0 and y_p[0] > t0:
            TN += 1
        elif y_t == 1 and y_p[1] > t1:
            TP += 1
        elif y_t == 0 and y_p[1] > t1:
            FP += 1
        elif y_t == 1 and y_p[0] > t0:
            FN += 1
        elif y_t == 0:  # and sample not classified
            FNN += 1
        else:   # y_p == 1 and sample not classified
            FNP += 1

    return {values.TP: TP, values.FP: FP, values.TN: TN, values.FN: FN, values.FNN: FNN, values.FNP: FNP}
