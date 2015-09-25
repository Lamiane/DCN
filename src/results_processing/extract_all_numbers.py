__author__ = 'nex'


def retrieve_lines_with_information(filename):
    with open(filename) as f:
        raw_results = f.read()

    informative_lines = []
    keep_adding = False
    for line in raw_results.split('\n'):
        if 'CV_ENTER' in line or 'layer name: h0' in line:
            informative_lines.append(line)
            keep_adding = True
            continue
        elif 'MODEL' in line and 'BEST MODEL' not in line:
            keep_adding = False
            continue
        elif 'NETWORK' in line or 'CV_MEAN' in line:
            informative_lines.append(line)
            continue
        elif 'Epochs' in line or 'learning_rate' in line or 'momentum' in line:
            informative_lines.append(line)
            continue
        elif 'F1Score1Threshold' in line:
            informative_lines.append(line)
            keep_adding = True
            continue
        elif 'corresponding threshold pair' in line:
            informative_lines.append(line)
            keep_adding = False
            continue
        elif '_ROC:' in line:
            informative_lines.append(line)
            keep_adding = True
            continue
        elif 'M_02' in line:
            informative_lines.append(line)
            keep_adding = False
            continue
        elif 'BEST MODEL' in line:
            keep_adding = True
        # the last statement
        if keep_adding:
            informative_lines.append(line)

    with open('processed', 'w') as my_file:
        for el in informative_lines:
            my_file.write(el+'\n')


def turn_into_structure(filename):
    with open(filename) as my_file:
        content = my_file.read()

    print 'content length', len(content)

    import re
    content = re.sub('f1score', 'fonescore', content)
    content = re.sub('1-ROC', 'ONE-ROC', content)

    from copy import deepcopy   # better safe than sorry!

    hyp_dict = {}
    cv_dicts_list = []
    cv_dict = {}
    models_list = []
    model_dict = {}
    epoch_list = []
    epoch_dict = {}

    finishing = False
    epoching = False
    youdening = False

    for line in content.split('\n'):
        if "BEST MODEL" in line:
            finishing = True
            continue
        elif finishing:
            hyp_dict['best_model'] = line
            hyp_dict['all_models'] = cv_dicts_list

        elif 'CV_ENTER' in line:
            cv_dict = {}
        elif 'CV_MEAN' in line:
            add_digit_to_dictionary(cv_dict, 'mean_score', line)
            cv_dict['models'] = deepcopy(models_list)
            models_list = []
            cv_dicts_list.append(deepcopy(cv_dict))
            cv_dict = {}
        elif 'SAMP' in line:
            cv_dict['architecture'] = line[6:-1]    # removing shit

        elif 'layer name: h0' in line:
            model_dict = {}
        elif '_ROC: Best roc score' in line:
            add_digit_to_dictionary(model_dict, 'best_score', line)
        elif '_ROC: Obtained for threshold:' in line:
            add_digit_to_dictionary(model_dict, 'threshold', line)
        elif 'precision' in line and not epoching:
            add_digit_to_dictionary(model_dict, 'precision', line)
        elif 'recall' in line and not epoching:
            add_digit_to_dictionary(model_dict, 'recall', line)
        elif 'fonescore' in line and not epoching:
            add_digit_to_dictionary(model_dict, 'f1score', line)
        elif 'M_02:' in line:
            model_dict['epochs'] = deepcopy(epoch_list)
            epoch_list = []
            models_list.append(deepcopy(model_dict))

        elif 'Epochs seen:' in line:
            epoch_dict = {}
            epoching = True
        elif epoching and ('corresponding threshold pair:' in line or 'y.shape' in line):
            epoching = False
            epoch_list.append(deepcopy(epoch_dict))
            epoch_dict = {}

        elif epoching:
            if 'learning_rate' in line:
                add_digit_to_dictionary(epoch_dict, 'learning_rate', line)
            elif 'momentum' in line:
                add_digit_to_dictionary(epoch_dict, 'momentum', line)
            elif 'Youden' in line:
                youdening = True
                continue
            elif youdening:
                if 'score' in line:
                    add_digit_to_dictionary(epoch_dict, 'Youden_score', line)
                elif 'threshold' in line:
                    add_digit_to_dictionary(epoch_dict, 'Youden_threshold', line)
                elif 'TP' in line:
                    add_two_digits_to_dictionary(epoch_dict, 'Youden_TP', 'Youden_TN', line)
                elif 'FP' in line:
                    add_two_digits_to_dictionary(epoch_dict, 'Youden_FP', 'Youden_FN', line)
                    epoch_dict['Youden_accuracy'] = compute_accuracy(epoch_dict['Youden_TP'], epoch_dict['Youden_TN'],
                                                                     epoch_dict['Youden_FP'], epoch_dict['Youden_FN'])
                elif 'precision' in line:
                    add_digit_to_dictionary(epoch_dict, 'Youden_precision', line)
                elif 'recall' in line:
                    add_digit_to_dictionary(epoch_dict, 'Youden_recall', line)
                elif 'fonescore' in line:
                    add_digit_to_dictionary(epoch_dict, 'Youden_f1score', line)
                elif '' == line:
                    youdening = False
    return hyp_dict


def add_digit_to_dictionary(dictionary, key, line_to_parse):
    from re import compile
    digit_finder = compile(r'(\d+\.\d+e-\d+)|(\d+\.\d+)|(\d+)')
    matched_digit = digit_finder.findall(line_to_parse)
    matched_list = []
    if matched_digit is not None:
        for tup in matched_digit:
            for el in tup:
                if el != '':
                    matched_list.append(el)
    if len(matched_list) == 1:
        matched_digit = matched_list[0]
    else:
        raise Exception('Things are going wrong in extract all numbers add digit to dictionary '+str(matched_list)
                        + '\n' + line_to_parse)
    if matched_digit is not None:
        dictionary[key] = float(matched_digit)


def add_two_digits_to_dictionary(dictionary, key1, key2, line_to_parse):
    from re import compile
    digit_finder = compile(r'(\d+\.\d+e-\d+)|(\d+\.\d+)|(\d+)')
    matched_digit = digit_finder.findall(line_to_parse)
    matched_list = []
    if matched_digit is not None:
        for tup in matched_digit:
            for el in tup:
                if el != '':
                    matched_list.append(el)
    if len(matched_list) != 2:
        raise Exception('Things are going wrong in extract all numbers add two digits to dictionary '+str(matched_list))
    if matched_digit is not None:
        dictionary[key1] = float(matched_list[0])
        dictionary[key2] = float(matched_list[1])


def compute_accuracy(tp, tn, fp, fn):
    # accuracy = (TP + TN) / TOTAL
    dividend = tp + tn
    if dividend == 0:
        accuracy = 0
    else:
        accuracy = float(dividend)/float(dividend + fp + fn)
    return accuracy


def check_sanity_of_dic(hyp_dict):
    only_once = False
    cv_dict_list = hyp_dict['all_models']
    assert len(cv_dict_list) == 20
    for idx, cv_dict in enumerate(cv_dict_list):
        print 'idex of cv_dict', idx
        models_list = cv_dict['models']
        temp = [mod['best_score'] for mod in models_list]
        assert len(models_list) == 5, str(temp)
        for single_model in models_list:
            all_epochs = single_model['epochs']
            assert len(all_epochs) > 0
            if only_once:
                print all_epochs[1]
    print 'passed part one'

    for idx, cv_dict in enumerate(cv_dict_list):
        print 'idex of cv_dict', idx
        models_list = cv_dict['models']
        best_from_list = []
        for models_dict in models_list:
            best_from_list.append(models_dict['best_score'])
        mean_from_dict = cv_dict['mean_score']
        from numpy import mean
        assert (1-mean_from_dict) - mean(best_from_list) < 0.05,\
            str(mean_from_dict) + "!= od mean po " + str(best_from_list) + \
            "\nroznica:" + str((1-mean_from_dict) - mean(best_from_list))
    print 'passed part two'
    return True


def testing():
    pass


def plot_global(hyp_dict, image_name, title):
    best_score = []
    cv_dict_list = hyp_dict['all_models']
    for cv_dict in cv_dict_list:
        best_score.append(cv_dict['mean_score'])
    print best_score
    create_plot([i+1 for i in range(len(best_score))], best_score, 'hyperopt iteration', 'mean score',
                image_name + '_global', title)


def create_plot(x, y, x_label, y_label, image_name, title):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(x, y, 'ro')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(image_name+'.png')
    plt.savefig(image_name+'.pdf')


def add_cv_dict_mean_score_to_plot(hyp_dict, plt):
    best_score = []
    cv_dict_list = hyp_dict['all_models']
    for cv_dict in cv_dict_list:
        best_score.append(cv_dict['mean_score'])
    x = [i+1 for i in range(len(best_score))]
    plt.plot(x, best_score)


def run_global(filename, title):
    m = turn_into_structure(filename)
    plot_global(m, filename, title)


def run_global_zusammen(filename_list, legend_list, filename_save):
    structures_list = []
    for filename in filename_list:
        structures_list.append(turn_into_structure(filename))
    import matplotlib.pyplot as plt
    for hyp_dict in structures_list:
        print 'd.keys', hyp_dict.keys()
        add_cv_dict_mean_score_to_plot(hyp_dict, plt)
    plt.legend(legend_list, loc='upper right')
    plt.xlabel('hyperopt iterations')
    plt.ylabel('mean score for given architecture')
    plt.savefig(filename_save+'.png')
    plt.savefig(filename_save+'.pdf')


def analyse_1(hyp_dict_MINIMUM):
    cv_dic_list = hyp_dict_MINIMUM['all_models']

    # minimum
    # upper = [1, 2, 4, 5, 9, 10, 12, 19]
    # lower = [0, 3, 6, 7, 11, 13, 14, 15, 16, 17, 18]

    # softmax
    upper = [6, 7, 8, 9, 10, 11, 12]
    lower = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 16, 17, 18, 19]

    import ast
    upper_architectures = []
    for index in upper:
        architecture_string = cv_dic_list[index]['architecture']
        dictios_tuple = ast.literal_eval(architecture_string)
        upper_architectures.append(dictios_tuple)

    lower_architectures = []
    for index in lower:
        architecture_string = cv_dic_list[index]['architecture']
        dictios_tuple = ast.literal_eval(architecture_string)
        lower_architectures.append(dictios_tuple)

    print 'upper h1', [dic_tup[1]['h1'] is None for dic_tup in upper_architectures]
    print 'lower h1', [dic_tup[1]['h1'] is None for dic_tup in lower_architectures]


def analyse_2(hyp_dict_MINIMUM):
    cv_dic_list = hyp_dict_MINIMUM['all_models']

    # minimum
    upper = [1, 2, 4, 5, 9, 10, 12, 19]
    lower = [0, 3, 6, 7, 11, 13, 14, 15, 16, 17, 18]

    # softmax
    # upper = [6, 7, 8, 9, 10, 11, 12]
    # lower = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 16, 17, 18, 19]

    from numpy import mean
    upper_besties = []
    for index in upper:
        models_list = cv_dic_list[index]['models']
        besties = []
        for models_dict in models_list:
            epoch_list = models_dict['epochs']
            best_youden_score = 0
            for epoch_dict in epoch_list:
                if epoch_dict['Youden_score'] > best_youden_score:
                    best_youden_score = epoch_dict['Youden_score']
            besties.append(best_youden_score)
        upper_besties.append(mean(besties))

    lower_besties = []
    for index in lower:
        models_list = cv_dic_list[index]['models']
        besties = []
        for models_dict in models_list:
            epoch_list = models_dict['epochs']
            best_youden_score = 0
            for epoch_dict in epoch_list:
                if epoch_dict['Youden_score'] > best_youden_score:
                    best_youden_score = epoch_dict['Youden_score']
            besties.append(best_youden_score)
        lower_besties.append(mean(besties))

    print 'upper besties', upper_besties
    print 'lower besties', lower_besties

    print 'upper besties mean', mean(upper_besties)
    print 'lower besties mean', mean(lower_besties)


def add_models_bests_scores_to_plot(hyp_dict, plt, change_color):
    from copy import deepcopy

    x_axis = []
    y_axis = []
    cv_dict_list = hyp_dict['all_models']
    for idx, cv_dict in enumerate(cv_dict_list):
        models_list = cv_dict['models']
        temp = []
        for model_dict in models_list:
            temp.append(model_dict['best_score'])
        x_axis.append(idx+1)
        y_axis.append(deepcopy(temp))

    a = plt.violinplot(dataset=tuple(y_axis), positions=x_axis, widths=1, showextrema=True, showmeans=True)
    if change_color:
        for pc in a['bodies']:
            pc.set_facecolor('blue')
            pc.set_color('blue')
            pc.set_edgecolor('blue')


def variance_zusammen(filename_list, legend_list, filename_save):
    structures_list = []
    for filename in filename_list:
        structures_list.append(turn_into_structure(filename))
    import matplotlib.pyplot as plt
    once = True
    for hyp_dict in structures_list:
        add_models_bests_scores_to_plot(hyp_dict, plt, once)
        once = False

    import matplotlib.patches as mpatches
    min_patch = mpatches.Patch(color='blue', label='minimum')
    softmax_patch = mpatches.Patch(color='yellow', label='softmax')
    plt.legend(handles=[min_patch, softmax_patch], loc='upper right')

    plt.xlabel('hyperopt iterations')
    plt.ylabel('best scores for given architecture')
    plt.savefig(filename_save+'.png')
    plt.savefig(filename_save+'.pdf')


def add_1_best(hyp_dict, plt):
    cv_dict_list = hyp_dict['all_models']
    plt.clf()

    # finding best architecture
    idx_best_score_tuple = (-1, 0)
    for idx, cv_dict in enumerate(cv_dict_list):
        if cv_dict['mean_score'] > idx_best_score_tuple[1]:
            idx_best_score_tuple = (idx, cv_dict['mean_score'])

    # plotting
    best_architecture_dict = cv_dict_list[idx_best_score_tuple[0]]
    for model_dict in best_architecture_dict['models']:
        model_scores = []

        for epoch_dict in model_dict['epochs']:
            model_scores.append(epoch_dict['Youden_score'])
        plt.plot([i for i in range(len(model_scores))], model_scores)


def learning_1_best(filename, filename_save, title):
    structured = turn_into_structure(filename)
    import matplotlib.pyplot as plt
    plt.clf()
    add_1_best(structured, plt)
    plt.title(title)
    plt.xlabel('epochs seen')
    plt.ylabel('Youden score')
    plt.savefig(filename_save+'.png')
    plt.savefig(filename_save+'.pdf')


def plot_max_in_epoch(hyp_dict, plt):
    cv_dict_list = hyp_dict['all_models']
    plt.clf()

    best_epoch = []
    for cv_dict in cv_dict_list:
        for model_dict in cv_dict['models']:
            epoch_score = [epoch['Youden_score'] for epoch in model_dict['epochs']]
            from numpy import argmax
            best_epoch.append(argmax(epoch_score))
    plt.hist(best_epoch, len(set(best_epoch)))


def max_in_epoch(filename, filename_save, title):
    structured = turn_into_structure(filename)
    import matplotlib.pyplot as plt
    plt.clf()
    plot_max_in_epoch(structured, plt)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('number of models that reached its maximum score')
    plt.savefig(filename_save+'.png')
    plt.savefig(filename_save+'.pdf')


def plot_number_of_epochs(hyp_dict, plt):
    cv_dict_list = hyp_dict['all_models']
    plt.clf()

    number_of_epochs = []
    for cv_dict in cv_dict_list:
        for model_dict in cv_dict['models']:
            number_of_epochs.append(len(model_dict['epochs'])-1)
    plt.hist(number_of_epochs, len(set(number_of_epochs)))


def number_of_epochs(filename, filename_save, title):
    structured = turn_into_structure(filename)
    import matplotlib.pyplot as plt
    plt.clf()
    plot_number_of_epochs(structured, plt)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('number of models that finished')
    plt.savefig(filename_save+'.png')
    plt.savefig(filename_save+'.pdf')
