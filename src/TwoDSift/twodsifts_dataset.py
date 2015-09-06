__author__ = 'igor'

import os
from copy import copy
import numpy as np
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils.rng import make_np_rng


class TwoDSiftData(DenseDesignMatrix):

    def __init__(self, filenames=[], y_val=[], nogap_type=True, labels=False, shuffle=True,
                 start=None, stop=None, cv=None, normal_run=True, indices_to_delete=None, shuffle_seed=1337,
        middle=[], middle_val=-1):
        # TODO remember filenames as a dictionary with additional information, like number of examples, etc.
        """
        :type filenames: list of data files to read
        :type y_val: list of output values associated with each of the files
        :type nogap_type: True if data written with no spaces between numbers
        :type labels: object
        :type shuffle: True if data is to be shuffled
        :type start: starting example
        :type stop: last example
        :type cv: list of CV parts [how many parts, [list of parts to use]]; default None
        """

        self.filenames = filenames

        if not os.path.isfile(self.filenames[0]):
            raise ValueError("Non-existent 2DSIFt file " + self.filenames[0])
        # get the protein name
        with open(self.filenames[0]) as f:
            line = f.readline()

        # checking if all the files are describing the same protein
        protein_names = []
        for file_name in filenames:
            with open(file_name) as opened_file:
                protein_names.append(opened_file.readline().split(":")[0])
        assert 1 == len(set(protein_names))
        del protein_names

        self.protein = line.split(":")[0]

        self.name = ""
        self.__residue_width = 9
        self.__residue_height = 6
        self.residues = 0
        self.receptors = []
        self.ligands = []
        self.start_residues = []
        self.name = self.protein
        # TODO perhaps window width and height should be parameters?
        self.__win_width = 0
        self.__win_height = 0
        self.batch_size = 0
        self.examples = 0
        self.cv = cv
        self.splits = []
        self.n_classes = 0
        self.remove_examples = False
        self.add_examples = not self.remove_examples
        self.shuffle_seed = shuffle_seed
        self.middle = middle
        self.middle_val = middle_val

        if y_val is None:
            raise ValueError('y_val must be provided')

        if nogap_type:
            topo_view, y, skipped = self.read_nogaps(filenames, y_val)
        else:
            raise NotImplementedError('read() function not implemented')

        # TODO check the assert together with number of skipped records
        # TODO labels are null for the time being

        self.n_classes = len(set(y))

        y = np.array(y).reshape((self.examples, 1))

        if indices_to_delete is not None:
            for tuple_element in reversed(indices_to_delete):
                from numpy import delete
                ind = [i for i in xrange(tuple_element[0], tuple_element[1]+1)]
                y = delete(y, ind)

        print 'y.shape', y.shape

        if shuffle:
            self.shuffle_data(topo_view, y)

        # if cv, then split shuffled data into cv parts, and then rearrange so that the parts to use are first
        # then limit the data to the part to be used
        # TODO raise(exception) in case of an error in parameters
        # TODO self.ligands should also be somehow restricted according to cv parameters
        if cv is not None and isinstance(cv, list) and self.is_cv_valid(self.cv):
            split_list = []     # indices on which split will be performed
            numberOfSplits = cv[0]
            print "topo view shape:", topo_view.shape
            fold_size = int(np.floor(float(topo_view.shape[0])/numberOfSplits))     # TODO: check
            remainder = np.remainder(topo_view.shape[0], numberOfSplits)            # TODO: check
            # populate split list
            for k in xrange(numberOfSplits+1):
                if len(split_list) > 0:
                    index = split_list[-1] + fold_size
                else:
                    index = 0
                if 0 < k <= remainder:
                    index += 1
                split_list.append(index)
            assert split_list[-1] == topo_view.shape[0]     # TODO: check

            indices_list = []   # list with indices that we want to leave
            # populate indices list
            for kth in cv[1]:
                _from = split_list[kth]
                _to = split_list[kth+1]
                indices_list.extend(xrange(_from, _to))

            topo_view = topo_view[indices_list, :, :, :]
            y = y[indices_list]
            self.examples = topo_view.shape[0]
        # /end cv part

        print 'topo_view.shape', topo_view.shape

        # TODO: id self.middle not empty - add middle to the dataset and shuffle it again
        if len(self.middle) > 0:
            topo_middle, y_middle, skipped = self.read_nogaps(middle, middle_val)
            topo_view = np.concatenate((topo_view, topo_middle))
            y = np.concatenate((y, y_middle))
            # self.examples was updated in read_nogaps

        # middle examples were just added at the end of topo_view, so it needs to be shuffled again
        if shuffle:
            self.shuffle_data(topo_view, y)

        # extending data
        if normal_run:
            print "wszedlem do normal_run"  # POCHA
            topo_view = self.preprocess_data(topo_view)
        print "WYSZEDLEM Z normal run"  # POCHA

        super(TwoDSiftData, self).__init__(topo_view=topo_view, y=y, axes=('b', 0, 1, 'c'), y_labels=self.n_classes)
        assert not np.any(np.isnan(self.X))

        print " PO SUPER"
        print "x shape", self.X.shape
        print "y shape", self.y.shape
        print 'examples:', self.examples

    def __str__(self):
        descr = "2D SiFT file:  " + self.name + \
                "\n\t" + "filename:  " + ", ".join(self.filenames) + \
                "\n\t" + "residues:  " + str(self.residues) + \
                "\n\t"
        descr += "ligands:   " + str(len(self.ligands)) + " : \n"
        descr += "\t" + "examples:        " + str(self.get_num_examples()) + \
                 "\n\t" + "topo axis:       " + str(self.get_topo_batch_axis()) + \
                 "\n\t" + "X.shape:         " + str(self.X.shape) + \
                 "\n\t" + "topo_view.shape: " + str(self.get_topological_view().shape) + \
                 "\n\t" + "n_classes:       " + str(self.n_classes) + \
                 "\n\t" + "cv:              " + str(self.cv) + \
                 "\n\t" + "splits   :       " + str(self.splits) + \
                 "\n\t" + "batch size :     " + str(self.batch_size)
        return descr

    def read_nogaps(self, filenames, y_val):
        """
        :rtype : ndarray
        """
        # first count number of lines in all the files
        file_line_counts = self.bufcount(filenames)
        topo_view = None
        y = []
        examples = 0
        ligands = []
        start_residues = []
        receptors = []
        skipped = []
        this_example_index = 0
        # TODO add reading from, and saving to, a pkl file
        for filename, current_y in zip(filenames, y_val):
            line_read = 0
            this_file_examples = 0
            this_file_skipped = []
            with open(filename) as f:
                for line in f:
                    line_read += 1
                    line = line.strip()
                    residues = line.split("(")

                    if topo_view is None:
                        first_residue = copy(residues[1]).strip()
                        first_residue = list(first_residue)[:-3]
                        if len(first_residue) % self.__residue_width != 0:
                            print self.__class__.__name__ + ": 2DSiFT width (" + str(len(first_residue)) + \
                                ") is not a multiple of " + "residue representation width (" + \
                                str(self.__residue_width) + "), line " + line_read
                            print os.path.basename(filename) + ":" + str(line_read) + " skip example"
                            this_file_skipped.append(line_read)
                            continue
                        assert len(first_residue) % self.__residue_width == 0, \
                            self.__class__.__name__ + ": 2DSiFT width (" + str(len(first_residue)) + \
                            ") is not a multiple of residue representation width (" + str(self.__residue_width) + ")"
                        residue_width = len(first_residue)
                        topo_view = np.zeros(((self.__residue_height * sum(file_line_counts)),
                                              len(first_residue)), dtype='float32')
                        this_example_view = topo_view[this_example_index:, :]
                        self.residues = this_example_view.shape[1] / self.__residue_width
                    else:
                        this_example_view = topo_view[this_example_index * self.__residue_height:, :]

                    for k, res in enumerate(residues[1:]):
                        res = list(res.strip())[:-3]
                        if len(res) % self.__residue_width != 0 or len(res) != residue_width:
                            this_example_view = None
                            break
                        this_example_view[k, :] = np.array(res, dtype='float32')

                    if this_example_view is None:
                        this_file_skipped.append(line_read)
                        continue
                    start = this_example_index * self.__residue_height

                    this_file_examples += 1
                    this_example_index += 1
                    receptors.append(residues[0].split(":")[0])
                    ligands.append(residues[0].split(":")[1])
                    start_residues.append(residues[0].split(":")[2])

                # add labelings for this set
                if len(current_y) == 1:
                    y.extend([current_y[0] for _ in range(this_file_examples)])
                else:
                    print "unimplemented multi labels"

                examples += this_file_examples
                skipped.append(this_file_skipped)
            # </open filename>
        # </for>

        # restrict the memory allocated to topo_view only to examples actually read
        topo_view = topo_view[:(self.__residue_height * examples), :]
        self.receptors = tuple(receptors)
        self.ligands = tuple(ligands)
        self.start_residues = tuple(start_residues)
        self.examples += examples
        topo_view = topo_view.reshape(examples,  # examples
                                      self.__residue_height,  # rows
                                      self.residues * self.__residue_width,  # columns
                                      1  # channels
                                      )
        y = np.array(y)
        return topo_view, y, skipped

    @property
    def window_shape(self):
        return self.__win_width, self.__win_height

    @property
    def input_shape(self):
        return self.get_topo_batch_axis()

    @window_shape.setter
    def window_shape(self, (width, height)):
        assert width > 0, "width has to have positive value"
        assert height > 0, "height has to have positive value"
        self.__win_width = width
        self.__win_height = height

    def get_window(self, residue, x_off, y_off):
        stwin = residue * self.__residue_width
        return self.data[stwin + x_off:stwin + x_off + self.__win_width, y_off:y_off + self.__win_height]

    @staticmethod
    def is_cv_valid(cv):
        if len(cv) != 2:
            print "cv split: incorrect cv parameter list:", cv
            return False
        if not isinstance(cv[0], int) or cv[0] < 2:
            print "cv split: incorrect number of parts to split:", cv[0]
            return False
        if len(cv[1]) == 1 and (cv[1][0] < 0 or cv[1][0] >= cv[0]):
            print "cv split: incorrect split to choose cv[1] =", cv[1][
                0], "in relation to number of splits cv[0] =", \
                cv[0]
            return False
        if len(cv[1]) == cv[0] - 1 and any([item < 0 or item >= cv[0] for item in cv[1]]):
            print "cv split: incorrect splits to choose cv[1] =", cv[1], "in relation to number of splits cv[0] =", \
                cv[0]
            return False
        if len(cv[1]) == cv[0] - 1 and len(cv[1]) > len(set(cv[1])):
            print "cv split: incorrect splits to choose, probably duplicates:", cv[1]
            return False
        return True

    def shuffle_data(self, topo_view, y):
        shuffle_rng = make_np_rng(None, default_seed=self.shuffle_seed, which_method="shuffle")
        for i in xrange(topo_view.shape[0]):
            j = shuffle_rng.randint(self.examples)
            # Copy ensures that memory is not aliased.
            tmp = topo_view[i, :, :, :].copy()
            topo_view[i, :, :, :] = topo_view[j, :, :, :]
            topo_view[j, :, :, :] = tmp

            tmp = y[i:i + 1].copy()
            y[i] = y[j]
            y[j] = tmp

    @staticmethod
    def preprocess_data(data):
        data_shape = list(data.shape)
        data_shape[1] *= 3      # extending size of new data
        data_shape[2] += 2*9    # should be parametrized? magic numbers! use self.window.width
        data_shape = tuple(data_shape)

        preprocessed_data = np.zeros(data_shape)

        for idx, sample in enumerate(data):
            preprocessed_data[idx, 0:6, 18:, :] = sample
            preprocessed_data[idx, 6:12, 9:-9, :] = sample
            preprocessed_data[idx, 12:, :-2*9, :] = sample  # magic numbers everywhere

        return preprocessed_data

    @staticmethod
    def bufcount(filenames=[]):
        lines = []
        for filename in filenames:
            f = open(filename)
            buf_size = 1024 * 1024
            read_f = f.read  # loop optimization
            this_file_lines = 0
            buf = read_f(buf_size)
            while buf:
                this_file_lines += buf.count('\n')
                buf = read_f(buf_size)
            lines.append(this_file_lines)

        return lines
