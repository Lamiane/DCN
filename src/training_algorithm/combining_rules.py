class CombineRule(object):
    def combine_dict(self, vec1_dict, vec2_dict):
        raise NotImplementedError

    # TODO: testing
    def zero_opposite_values_dict(self, vec1_dict, vec2_dict):
        # input parameters are dictionaries with numpy arrays
        # copy the input vectors in order not to change them in any way
        # this might cause OutOfMemory problems
        # but it helps avoiding difficult to track changes in network parameters
        import copy
        vec1_dict_copy = copy.deepcopy(vec1_dict)
        vec2_dict_copy = copy.deepcopy(vec2_dict)

        # for each numpy array in the dictionaries
        for key in vec1_dict_copy:
            # check if keys are valid
            if key not in vec2_dict_copy:
                raise KeyError
            # getting indices where values have opposite sign
            indices_to_zero = (vec1_dict_copy[key] * vec2_dict_copy[key]) < 0
            vec1_dict_copy[key][indices_to_zero] = 0
            vec2_dict_copy[key][indices_to_zero] = 0
        return vec1_dict_copy, vec2_dict_copy


class Minimum(CombineRule):
    # TODO: testing
    # TODO: check if it started working
    def combine_dict(self, vec1_dict, vec2_dict):
        import numpy as np
        # input parameters are dictionaries with numpy arrays
        # copy the input vectors in order not to change them in any way
        # this might cause OutOfMemory problems
        # but it helps avoiding difficult to track changes in network parameters
        import copy
        vec1_dict_zeroed, vec2_dict_zeroed = self.zero_opposite_values_dict(copy.deepcopy(vec1_dict),
                                                                            copy.deepcopy(vec2_dict))

        minimum_dict = {}
        # for each numpy array
        for key in vec1_dict_zeroed:
            # check if keys are valid
            if key not in vec2_dict_zeroed:
                raise KeyError
            # TODO: testing
            sign = np.sign(vec1_dict_zeroed[key])
            # element-wise multiplication
            minimum_dict[key] = np.multiply(sign,
                                            np.minimum(np.abs(vec1_dict_zeroed[key]), np.abs(vec2_dict_zeroed[key]))
                                            )
            # minimum_dict[key] = -1 razy te indices, jakos to ogarnac

        return minimum_dict


class Mean(CombineRule):
    # TODO: testing
    def combine_dict(self, vec1_dict, vec2_dict):
        # input parameters are dictionaries with numpy arrays
        # importing numpy for later use
        import numpy as np
        # copy the input vectors in order not to change them in any way
        # this might cause OutOfMemory problems
        # but it helps avoiding difficult to track changes in network parameters
        import copy
        vec1_dict_zeroed, vec2_dict_zeroed = self.zero_opposite_values_dict(copy.deepcopy(vec1_dict),
                                                                            copy.deepcopy(vec2_dict))
        mean_dict = {}
        for key in vec1_dict_zeroed:
            if key not in vec2_dict_zeroed:
                raise KeyError
            mean_dict[key] = (vec1_dict_zeroed+vec2_dict_zeroed)/2

        return mean_dict


class Softmax(CombineRule):
    # TODO: testing
    def combine_dict(self, vec1_dict, vec2_dict):
        import numpy as np
        # copy the input vectors in order not to change them in any way
        # this might cause OutOfMemory problems
        # but it helps avoiding difficult to track changes in network parameters
        import copy
        vec1_dict_zeroed, vec2_dict_zeroed = self.zero_opposite_values_dict(copy.deepcopy(vec1_dict),
                                                                            copy.deepcopy(vec2_dict))

        softmax_dict = {}
        for key in vec1_dict_zeroed:
            if key not in vec2_dict_zeroed:
                raise KeyError
            ve1 = vec1_dict_zeroed[key]
            ve2 = vec2_dict_zeroed[key]
            divisor = np.exp(ve1) + np.exp(ve2)
            softmax_dict[key] = np.multiply(ve1, np.exp(ve1)/divisor) + np.multiply(ve2, np.exp(ve2)/divisor)

        return softmax_dict
