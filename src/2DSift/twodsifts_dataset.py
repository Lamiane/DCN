__author__ = 'igor'

# todo: zoptymalizowac improty
import os
import time
from copy import copy
import numpy as np
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.utils.rng import make_np_rng
import utils


class TwoDSiftData(DenseDesignMatrix):
    def __init__(self, filenames=[], y_val=[], nogap_type=True, replicate=0, labels=False, shuffle=True,
                 start=None, stop=None, cv=None):
        # TODO remember filenames as a dictionary with additional information, like number of examples, etc.
        """
        :type filenames: list of data files to read
        :type y_val: list of output values associated with each of the files
        :type nogap_type: True if data written with no spaces between numbers
        :type replicate: number of extra replications of each example; default 0
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
        self.protein = line.split(":")[0]
        # TODO:@pocha nie powinnysmy sie upewniac, ze wszystkie pliki sa od tego samego bialka?

        self.name = ""
        self.__residue_width = 9
        self.__residue_height = 6
        self.residues = 0
        self.receptors = []
        self.ligands = []
        self.start_residues = []
        self.replicate = replicate
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

        if nogap_type:
            # TODO add a skipped attribute to be returned
            # TODO @pocha: isn't it done already?
            topo_view, y, skipped = self.read_nogaps(filenames, y_val)
        else:
            # topo_view = self.read()
            print 'read() function for gap_type files is deprecated, please write a new one to use this file' \
                  'or use a file with no gaps'
            return
        # TODO check the assert together with number of skipped records

        # TODO labels are null for the time being
        # TODO @pocha: are they?
        self.n_classes = len(set(y))

        if y is not None:
            y = np.array(y).reshape((self.examples, 1))
        else:
            # TODO tu liczba generowanych etykiet jest ustalona na 2, a powinna byc taka jak w parametrach
            # TODO @pocha ja bym to wywalila, no bo no serio...
            y = np.random.randint(0, 2, (self.examples, 1))

        if shuffle:
            self.shuffle_data(topo_view, y)

        # if cv, then split shuffled data into cv parts, and then rearrange so that the parts to use are first
        # then limit the data to the part to be used
        # TODO raise(exception) in case of an error in parameters
        # TODO self.ligands should also be somehow restricted according to cv parameters
        if cv is not None and isinstance(cv, list) and self.is_cv_valid(self.cv):
            # compute the possible splits
            self.splits, remove_ans, add_ans = utils.splits(self.examples, cv[0])
            # check the splits returned and perform some additional example removal / addition needed
            if remove_ans[1] == self.examples:
                self.batch_size = remove_ans[0]
            else:
                # addition/removal of examples is needed to have a good batch size
                if self.remove_examples:
                    # cut out the last examples, provided they were shuffle earlier
                    # TODO add some option if examples are not shuffled
                    topo_view = topo_view[:remove_ans[1], :, :, :]
                    y = y[:remove_ans[1]]
                    self.splits = remove_ans[2]
                    self.batch_size = remove_ans[0]
                    self.examples = remove_ans[1]
                elif self.add_examples:
                    topo_view = np.vstack((topo_view, topo_view[add_ans[3], :, :, :]))
                    y = np.vstack((y, y[add_ans[3]]))
                    self.splits = add_ans[2]
                    self.batch_size = add_ans[0]
                    self.examples = add_ans[1]
                else:
                    raise ValueError("self.remove_examples / self.add_examples not set")
            # cutting into cross validation sets
            if len(cv[1]) == 1:
                if cv[1][0] == cv[0] - 1:
                    from_example = self.splits[-1]
                    first_not_example = self.examples
                elif cv[1][0] == 0:
                    from_example = 0
                    first_not_example = self.splits[1]
                else:
                    from_example = self.splits[cv[1][0]]
                    first_not_example = self.splits[cv[1][0] + 1]
                topo_view = topo_view[from_example:first_not_example, :, :, :]
                y = y[from_example:first_not_example]
            else:
                cv[1] = sorted(cv[1])
                missing = list(set(range(cv[0])) - set(cv[1]))[0]
                if missing == cv[0] - 1:
                    from_example = 0
                    first_not_example = self.splits[-1]
                    topo_view = topo_view[from_example:first_not_example, :, :, :]
                    y = y[from_example:first_not_example]
                elif missing == 0:
                    from_example = self.splits[1]
                    first_not_example = self.examples
                    topo_view = topo_view[from_example:first_not_example, :, :, :]
                    y = y[from_example:first_not_example]
                else:
                    from_example_first = 0
                    first_not_example_first = self.splits[missing]
                    from_example_second = self.splits[missing + 1]
                    first_not_example_second = self.examples
                    topo_view = np.vstack((topo_view[from_example_first:first_not_example_first, :, :, :],
                                           topo_view[from_example_second:first_not_example_second, :, :, :]))
                    y = np.vstack((y[from_example_first:first_not_example_first],
                                   y[from_example_second:first_not_example_second]))

            start = 0
            stop = topo_view.shape[0]
            self.examples = stop - start
        # /end cv part

        super(TwoDSiftData, self).__init__(topo_view=topo_view, y=y, axes=('b', 0, 1, 'c'), y_labels=self.n_classes)
        assert not np.any(np.isnan(self.X))

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop=' + str(stop) + '>' +
                                 'm=' + str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop, :]
            self.examples = stop - start + 1
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                                 % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop, :]
            else:
                self.y = self.y[start:stop]
            assert self.y.shape[0] == stop - start

        self.X = self.X / np.abs(self.X).max()


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
        file_line_counts = utils.bufcount(filenames)
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
                        topo_view = np.zeros(((self.__residue_height * sum(file_line_counts) * (self.replicate + 1)),
                                              len(first_residue)), dtype='float32')
                        this_example_view = topo_view[this_example_index:, :]
                        self.residues = this_example_view.shape[1] / self.__residue_width
                    else:
                        this_example_view = topo_view[
                                            this_example_index * self.__residue_height * (self.replicate + 1):, :]

                    for k, res in enumerate(residues[1:]):
                        res = list(res.strip())[:-3]
                        if len(res) % self.__residue_width != 0 or len(res) != residue_width:
                            this_example_view = None
                            break
                        this_example_view[k, :] = np.array(res, dtype='float32')

                    if this_example_view is None:
                        this_file_skipped.append(line_read)
                        continue
                    start = this_example_index * self.__residue_height * (self.replicate + 1)

                    for repl_ind in range(self.replicate):
                        this_example_start = start + (repl_ind + 1) * self.__residue_height
                        topo_view[this_example_start:this_example_start + self.__residue_height, :] = \
                            topo_view[start:start + self.__residue_height, :]

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
        topo_view = topo_view[:(self.__residue_height * examples * (self.replicate + 1)), :]
        self.receptors = tuple(receptors)
        self.ligands = tuple(ligands)
        self.start_residues = tuple(start_residues)
        self.__residue_height += self.replicate * self.__residue_height
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

    def is_cv_valid(self, cv):
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
        self.shuffle_rng = make_np_rng(None, [1, 2, 3], which_method="shuffle")
        for i in xrange(topo_view.shape[0]):
            j = self.shuffle_rng.randint(self.examples)
            # Copy ensures that memory is not aliased.
            tmp = topo_view[i, :, :, :].copy()
            topo_view[i, :, :, :] = topo_view[j, :, :, :]
            topo_view[j, :, :, :] = tmp

            tmp = y[i:i + 1].copy()
            y[i] = y[j]
            y[j] = tmp
