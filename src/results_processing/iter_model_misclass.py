__author__ = 'agnieszka'


def process_raw_data(pathname):
    import subprocess
    command1 = "cat " + pathname
    s1 = subprocess.Popen(command1.split(), stdout=subprocess.PIPE)
    ppp = s1.communicate()[0].split('\n')

    # not as I planned but it works well
    processed_data = ""
    for line in ppp:
        # if "misclass" in line or "ITERATION" in line:
        if "misclass" in line or "SAMP" in line:
            processed_data = processed_data + "\n" + line
    return processed_data


def prepare_data_for_plotting(processed_data):
    from re import compile
    processed_data = processed_data.split('\n')

    results = []
    idx = -1

    # processing the file
    digit_finder = compile(r'\d+.\d+')

    for line in processed_data:
        # either create a list for a new experiment
        # if 'ITERATION' in line:
        if 'SAMP' in line:
            idx += 1
            results.append([])
        # or add a new misclass error to the experiment now being processed
        else:
            matched = digit_finder.search(line)
            if matched is not None:
                results[idx].append(float(matched.group()))

    return results


def make_plots(y, min_n_of_epochs):
    import matplotlib.pyplot as plt
    from os.path import join, exists
    from os import makedirs
    import sys
    sys.path.append('..')
    import configuration.model as config
    from utils.common import get_timestamp
    timestamp = get_timestamp()
    # we are creating directory in which all the plots will be stored
    storing_directory = join(config.path_for_storing, str(timestamp)+"_iter_middle_misclass")
    # if directory doesn't exist (which should be always) create it
    if not exists(storing_directory):
        makedirs(storing_directory)
    # for each experiment that has been run
    for current_idx, experiment in enumerate(y):
        # if it has seen at least 3 epochs
        if len(experiment) > min_n_of_epochs:
            # save plot of the misclass error w.r.t number of epochs seen
            plt.plot(range(1, len(experiment)+1), experiment, 'r-')
            plt.xlabel("number of epochs seen")
            plt.ylabel("misclass error")
            plt.savefig(join(storing_directory, 'epochs_misclass_model'+str(current_idx)+".png"),
                        bbox_inches='tight')
            plt.clf()


def print_additional_information(y, min_n_of_epochs):
    from numpy import mean
    minimums = []
    # for each experiment that has been run
    for experiment in y:
        # if it has seen at least 3 epochs
        if len(experiment) > min_n_of_epochs:
            # find minimal misclass error for this experiment
            minimums.append(min(experiment))

    print "minimums:", minimums
    print "maximal minimum:", max(minimums)
    print "minimal minimum:", min(minimums)
    print "mean", mean(minimums)


def process(raw_data=None, preprocessed_data=None, plots=True, additional_output=True, min_n_of_epochs=2):
    # xoring raw_data and preprocessed_data
    # TODO: maybe assert will be better?
    if raw_data is None and preprocessed_data is None:
        raise ValueError("Pathname to either raw data either preprocessed data should be provided")
    if raw_data is not None and preprocessed_data is not None:
        raise ValueError("Pathname to either raw data either preprocessed data should be provided")

    if raw_data is not None:
        preprocessed_data = process_raw_data(raw_data)
    y = prepare_data_for_plotting(preprocessed_data)

    if plots:
        make_plots(y, min_n_of_epochs=min_n_of_epochs)

    if additional_output:
        print_additional_information(y, min_n_of_epochs=min_n_of_epochs)
