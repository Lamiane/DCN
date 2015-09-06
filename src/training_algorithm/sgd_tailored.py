from pylearn2.training_algorithms.sgd import SGD
import numpy as np
from pylearn2.space import CompositeSpace
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping
from blessings import Terminal
t = Terminal()


class SgdTailored(SGD):

    def __init__(self, learning_rate, combine_updates_rule, cost=None, batch_size=None,
                 monitoring_batch_size=None, monitoring_batches=None,
                 monitoring_dataset=None, monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None, learning_rule=None,
                 set_batch_size=False, train_iteration_mode=None, batches_per_iter=None,
                 theano_function_mode=None, monitoring_costs=None, seed=[2012, 10, 5]):

        self.combine_updates_rule = combine_updates_rule
        self.debug_dict = {}
        self.second = False
        super(SgdTailored, self).__init__(learning_rate=learning_rate, cost=cost, batch_size=batch_size,
                                          monitoring_batch_size=monitoring_batch_size,
                                          monitoring_batches=monitoring_batches, monitoring_dataset=monitoring_dataset,
                                          monitor_iteration_mode=monitor_iteration_mode,
                                          termination_criterion=termination_criterion, update_callbacks=update_callbacks,
                                          learning_rule=learning_rule, set_batch_size=set_batch_size,
                                          train_iteration_mode=train_iteration_mode, batches_per_iter=batches_per_iter,
                                          theano_function_mode=theano_function_mode, monitoring_costs=monitoring_costs,
                                          seed=seed)

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

        # print 'space tuple', type(space_tuple), space_tuple
        from pylearn2.space import VectorSpace

        ###############################################
        # # # CHANGINGS TO THE ORIGINAL ALGORITHM # # #
        ###############################################

        # we have 3 classes in dataset (active, inactive, middle), but only two softmax neurons
        # therefore VectorSpace has dim = 2 and an error will be raised when trying to convert
        # label to a vector of length 2. So we change the vector length for a while and convert
        # things manually.
        space_tuple = (space_tuple[0], VectorSpace(dim=3))

        #############################
        # # #  END OF CHANGINGS # # #
        #############################

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

        # print 'flat data specs', type(flat_data_specs), flat_data_specs
        # flat data specs <type 'tuple'>
        # (CompositeSpace(Conv2DSpace(shape=(18, 3492), num_channels=1, axes=('c', 0, 1, 'b'), dtype=float64),
        #                             VectorSpace(dim=2, dtype=float64)),
        #                 'features', 'targets'))

        on_load_batch = self.on_load_batch
        for batch in iterator:
            # batch is a list with two numpy arrays: [sample, label]
            # self.params is a list with theano.tensor.sharedvar.TensorSharedVariables
            # theano.tensor.sharedvar.TensorSharedVariable.get_value() returns numpy.array
            # you can set value with theano.tensor.sharedvar.TensorSharedVariable.set_value(np.array_object)

            # this being here might cause troubles as batch is a nasty thing right now
            for callback in on_load_batch:
                callback(*batch)

            ###############################################
            # # # CHANGINGS TO THE ORIGINAL ALGORITHM # # #
            ###############################################

            self.print_params("on entering iteration", t.cyan)

            # GOOD ADVICE: if something is very wrong check it the following map is valid
            # TODO: check this
            # active     1    [[ 0. 1. 0. ]]    [[ 0. 1. ]]
            # nonactive  0    [[ 1. 0. 0. ]]    [[ 1. 0. ]]
            # middle    -1    [[ 0. 0. 1. ]]

            batch_1_on_load = batch[1].copy()

            # if label was '0'
            if (batch[1] == np.array((1, 0, 0))).all():
                # print "example: nonactive"
                batch = (batch[0], np.reshape(np.array((1, 0)), (1, 2)))
                self.sgd_update(*batch)
            # if label was '1'
            elif (batch[1] == np.array((0, 1, 0))).all():
                # print "example: active"
                batch = (batch[0], np.reshape(np.array((0, 1)), (1, 2)))
                self.sgd_update(*batch)
            # else we have to deal with unlabeled example
            else:
                # print "example: middle"
                parameters_on_load = self.get_parameters()

                ######################################
                # # # RUNNING AS INACTIVE SAMPLE # # #
                ######################################
                # print 'running as inactive'
                # setting label as inactive
                batch = (batch[0], np.reshape(np.array((1, 0)), (1, 2)))
                self.print_params("on entering inactive", t.blue)
                # updating the model
                self.sgd_update(*batch)
                self.print_params("after update inactive", t.green)
                # remember changing in parameters
                params_after_inactive = self.get_parameters()
                diff_inactive = self.get_difference(parameters_on_load, params_after_inactive)
                self.print_dict_of_params(diff_inactive, "difference")
                # bring back on load parameters
                self.restore_parameters(parameters_on_load)
                self.print_params('after restore', t.yellow)
                ####################################
                # # # RUNNING AS ACTIVE SAMPLE # # #
                ####################################
                # print 'running as active'
                # setting label as active
                batch = (batch[0], np.reshape(np.array((0, 1)), (1, 2)))
                self.print_params('on entering active', t.blue)
                # updating the model
                self.sgd_update(*batch)
                self.print_params('after update active', t.green)
                # remember changing in parameters
                params_after_active = self.get_parameters()
                diff_active = self.get_difference(parameters_on_load, params_after_active)
                self.print_dict_of_params(diff_active, "difference")
                # bring back on load parameters
                self.restore_parameters(parameters_on_load)
                self.print_params('after restore', t.yellow)
                ##############################
                # # # UPDATING THE MODEL # # #
                ##############################
                update_vector = self.calculate_update(diff_active, diff_inactive)
                self.print_dict_of_params(update_vector, "update vector")
                self.update_non_classification_parameters(update_vector)

            # end of if

            self.print_params('on leaving', t.red)

            # iterator might return a smaller batch if dataset size
            # isn't divisible by batch_size
            # Note: if data_specs[0] is a NullSpace, there is no way to know
            # how many examples would actually have been in the batch,
            # since it was empty, so actual_batch_size would be reported as 0.

            # OK, now lines below need batch in the previous size. So I just set the batch to what is used to be
            # before my wicked transformations.
            batch = (batch[0], batch_1_on_load)

            self.print_self_debug()

            if self.second:
                import sys
                sys.exit(0)

            #############################
            # # #  END OF CHANGINGS # # #
            #############################
            actual_batch_size = flat_data_specs[0].np_batch_size(batch)
            self.monitor.report_batch(actual_batch_size)
            for callback in self.update_callbacks:
                callback(self)

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if not isfinite(value):
                raise Exception("NaN in " + param.name)
        self.second = True

    def get_parameters(self):
        param_dict = {}
        for param in self.params:
            param_dict[param.name] = param.get_value().copy()
        return param_dict

    def restore_parameters(self, saved_parameters_dict):
        for param in self.params:
            param.set_value(saved_parameters_dict[param.name].copy())

    def get_difference(self, base_value_dict, new_value_dict):
        difference_dict = {}
        for key in new_value_dict:
            if key not in base_value_dict:
                raise KeyError(key+"is not in base_value_dict")
            difference_dict[key] = base_value_dict[key]-new_value_dict[key]
        return difference_dict

    def calculate_update(self, vec1_dict, vec2_dict):
        return self.combine_updates_rule.combine_dict(vec1_dict, vec2_dict)

    def update_non_classification_parameters(self, update_vector_dict):
        # bez updejtu czesci klasyfikacyjnej!
        # if 'softmax' in param.name or 'classif' in param.name
        for param in self.params:
            # not updating the classification part of network
            # ASSUMPTION: layers responsible for classification have 'softmax' or 'classif' in name
            # i.e. they are called for example softmax_1, classify, 01_classification_layer...
            if 'softmax' in param.name or 'classif' in param.name:
                continue
            # else
            new_value = param.get_value() + update_vector_dict[param.name].copy()
            param.set_value(new_value)

    def print_params(self, information, terminal_configuration, step_by_step=False, param_by_param=True):
        if step_by_step:
            # print information.upper()
            for param in self.params:
                # print terminal_configuration + param.name, np.ravel(param.get_value())[0:9], t.normal + '\n'
                pass
        elif param_by_param:
            for param in self.params:
                if param.name not in self.debug_dict:
                    self.debug_dict[param.name] = ''
                self.debug_dict[param.name] += information.upper() + ' ' + terminal_configuration +\
                    str(np.ravel(param.get_value())[0:9]) + ' ' + t.normal + '\n'

    def print_dict_of_params(self, dict_of_params, information, step_by_step=False, param_by_param=True):
        if step_by_step:
            # print information.upper()
            for key in dict_of_params:
                # print key, np.ravel(dict_of_params[key])[0:9], '\n'
                pass
        elif param_by_param:
            for key in dict_of_params:
                if key not in self.debug_dict:
                    self.debug_dict[key] = ''
                self.debug_dict[key] += information.upper() + ' ' + str(np.ravel(dict_of_params[key])[0:9]) + '\n'

    def print_self_debug(self, clear_self_debug_dict=True):
        for key in self.debug_dict:
            # print key, self.debug_dict[key], '\n'
            pass
        if clear_self_debug_dict:
            self.debug_dict = {}
