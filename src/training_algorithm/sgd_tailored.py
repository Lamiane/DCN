from pylearn2.training_algorithms.sgd import SGD
import numpy as np
from pylearn2.space import CompositeSpace
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping


class SgdTailored(SGD):


    # we only want to change a part of training algorithm
    def train(self, dataset):
        """
        Runs one epoch of SGD training on the specified dataset.

        Parameters
        ----------
        dataset : Dataset
        """
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if not isfinite(value):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None

        data_specs = self.cost.get_data_specs(self.model)

        # The iterator should be built from flat data specs, so it returns
        # flat, non-redundent tuples of data.
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
        if len(space_tuple) == 0:
            # No data will be returned by the iterator, and it is impossible
            # to know the size of the actual batch.
            # It is not decided yet what the right thing to do should be.
            raise NotImplementedError(
                "Unable to train with SGD, because "
                "the cost does not actually use data from the data set. "
                "data_specs: %s" % str(data_specs))
        flat_data_specs = (CompositeSpace(space_tuple), source_tuple)

        iterator = dataset.iterator(mode=self.train_iteration_mode,
                                    batch_size=self.batch_size,
                                    data_specs=flat_data_specs,
                                    return_tuple=True, rng=rng,
                                    num_batches=self.batches_per_iter)

        on_load_batch = self.on_load_batch
        for batch in iterator:
            for callback in on_load_batch:
                callback(*batch)
            ###############################################
            # # # CHANGINGS TO THE ORIGINAL ALGORITHM # # #
            ###############################################

            from blessings import Terminal
            t = Terminal()
            from pprint import pprint

            print t.bold_cyan('\n\n\n\tPARAMS before update:')
            for param in self.params:
                print param  #, param.get_value().shape
                pprint(np.ravel(param.get_value())[0:9])

            print 'batch[1]', type(batch[1]), batch[1]
            condition = (batch[1] == np.array([[1., 0.]])).all()
            print t.bold_red(str(condition))

            if condition:
                temp4 = self.params[4].get_value().copy()
                temp5 = self.params[5].get_value().copy()

#            if condition:
#                print 'before'
#                print t.bold_green(str(temp4[0:9]))
#                print t.bold_magenta(str(temp5[0:9]))

            self.sgd_update(*batch)

            print t.bold_cyan('\n\tPARAMS after update:')
#            if condition:
#                print 'after'
#                print t.bold_green(str(temp4[0:9]))
#                print t.bold_magenta(str(temp5[0:9]))

            if condition:
                 self.params[4].set_value(temp4)
                 self.params[5].set_value(temp5)

            for param in self.params:
                print param  #, param.get_value().shape
                print np.ravel(param.get_value())[0:9]

            #############################
            # # #  END OF CHANGINGS # # #
            #############################

            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if not isfinite(value):
                raise Exception("NaN in " + param.name)