__author__ = 'agnieszka'
from re import compile


def is_valid_attribute(name):
    pattern = compile("[A-Za-z]")    # nie bierzemy zadnych metod ani pol ktore zaczynaja sie od __
    if pattern.match(name) is None:
        return False
    return True


def parse_to_yaml(obj, tabulators=0):
    tabs = 4*' '
    output = tabulators*tabs
    if type(obj) is type(None):
        return 'None'    # this shall never happen, how to deal with it? raise an exception?
    elif type(obj) in (type([]), type(())):
        output += "["
        for element in obj:
            output += parse_to_yaml(element, tabulators+1)
        output += "],"
    elif type(obj) is type({}):
        output += "{"
        for keys, vals in obj.iteritems():
            output += (tabulators+1)*tabs + keys + ": " + parse_to_yaml(vals, 0)
        output += tabulators*tabs + "},\n"
    elif isinstance(obj, str):
        output = obj.__str__() + ","
    elif type(obj) in [type(0), type(0.0), type(True)]:
        output = obj.__repr__() + ", "
    else:
        list_of_parameters = filter(is_valid_attribute, dir(obj))
        output = "!obj:" + obj.__hierarchy__ + " {"
        for element in list_of_parameters:
            val = getattr(obj, element)
            if val is not None:
                output += "\n" + (tabulators+1)*tabs + element + ": " + parse_to_yaml(val, tabulators+1)
        output += '\n' + tabulators*tabs + '},\n'

    return output


# TODO: it's easy to write it in more general form
def parse_weight_decay(mod):
    weight_decay_coeffs = "'h0': 0.00005,"
    if len(mod.layers) == 3:
        weight_decay_coeffs += "\n'h1': 0.00005,"
    weight_decay_coeffs += "\n" + "'softmax': 0.00005"
    return weight_decay_coeffs
