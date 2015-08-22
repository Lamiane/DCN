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

        print 'space tuple', type(space_tuple), space_tuple
        from pylearn2.space import VectorSpace
        space_tuple = (space_tuple[0], VectorSpace(dim=3))

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

        print 'flat data specs', type(flat_data_specs), flat_data_specs
        # flat data specs <type 'tuple'>
        # (CompositeSpace(Conv2DSpace(shape=(18, 3492), num_channels=1, axes=('c', 0, 1, 'b'), dtype=float64),
        #                             VectorSpace(dim=2, dtype=float64)),
        #                 'features', 'targets'))

        on_load_batch = self.on_load_batch
        for batch in iterator:
            # this being here might cause troubles as batch is a nasty thing right now
            for callback in on_load_batch:
               callback(*batch)

            ###############################################
            # # # CHANGINGS TO THE ORIGINAL ALGORITHM # # #
            ###############################################

            # GOOD ADVICE: if something is very wrong check it the following map is valid
            # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
            # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
            # middle    -1    [[ 0. 0. 1. ]]

            # if label was '0'
            if (batch[1] == np.array((1, 0, 0))).all():
                batch = (batch[0], np.array((1, 0)))
                self.run_normal(dataset, batch)
            # if label was '1'
            elif (batch[1] == np.array((0, 1, 0))).all():
                batch = (batch[0], np.array((0, 1)))
                self.run_normal((dataset, batch))
            # else we have to deal with unlabeled example
            else:
                parameters_on_load = self.get_parameters()
                # running for active (or nonactive?)
                batch = (batch[0], np.array((1, 0)))
                self.run_normal(dataset, batch)
                diff1 = self.get_parameters()
                self.restore_parameters(parameters_on_load)
                # running for nonactive (or active?)
                batch = (batch[0], np.array((0, 1)))
                self.run_normal(dataset, batch)
                diff2 = self.get_parameters()
                self.restore_parameters(parameters_on_load)
                # updating the model
                update_vector = self.calculate_update(diff1, diff2)
                self.update_parameters(update_vector)

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

    def run_normal(self, dataset, batch):
        pass

    def get_parameters(self):
        pass

    def restore_parameters(self, saved_parameters):
        pass

    def calculate_update(self, vec1, vec2):
        pass

    def update_parameters(self, update_vector):
        pass

