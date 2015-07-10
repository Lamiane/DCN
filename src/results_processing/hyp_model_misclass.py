__author__ = 'agnieszka'


def process_raw_data(pathname):
    import subprocess
    command1 = "cat " + pathname
    command2 = "grep M_01 "
    s1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)
    s2 = subprocess.Popen(command2.split(), stdin=s1.stdout, stdout=subprocess.PIPE)
    processed_data = s2.communicate()[0]
    return processed_data


def prepare_data_for_plotting(processed_data):
    from re import compile
    processed_data = processed_data.split('\n')
    digit_finder = compile(r'\d+.\d+')

    results = []
    for line in processed_data:
        matched = digit_finder.search(line)
        if matched is not None:
            try:
                results.append(float(matched.group()))
            except ValueError:
                print line
    return results


def make_plots(y):
    import matplotlib.pyplot as plt
    from os.path import join
    import sys
    sys.path.append('..')
    import configuration.model as config
    from utils.common import get_timestamp
    timestamp = get_timestamp()
    plt.plot(range(1, len(y)+1), y, 'r-')
    plt.xlabel("number of model seen by hyperopt")
    plt.ylabel("misclass error")
    plt.savefig(join(config.path_for_storing, timestamp+'hyp_model_misclass.png'), bbox_inches='tight')
    plt.clf()


def print_additional_information(y):
    minimal = min(y)
    print 'Minimal misclass error was:', minimal


def process(raw_data=None, preprocessed_data=None, plots=True, additional_output=True):
    # xoring raw_data and preprocessed_data
    # TODO: maybe assert will be better?
    if raw_data is None and preprocessed_data is None:
        raise ValueError("Pathname to either raw data either preprocessed data should be provided")
    if raw_data is not None and preprocessed_data is not None:
        raise ValueError("Pathname to either raw data either preprocessed data should be provided")

    if raw_data is not None:
        print "in in "
        preprocessed_data = process_raw_data(raw_data)

    y = prepare_data_for_plotting(preprocessed_data)

    if plots:
        make_plots(y)

    if additional_output:
        print_additional_information(y)