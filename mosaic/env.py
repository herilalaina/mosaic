"""Base environement class."""

import time
import datetime
import numpy as np


class AbstractEnvironment:
    def __init__(self, seed=42):
        """Abstract class for environment

        :param seed: random seed
        """
        self.seed = seed
        self.start_time = time.time()
        self.rng = np.random.RandomState(seed)

    def rollout(self, history=[]):
        """Rollout method to generate complete configuration starting with `history`

        :param history: current incomplete configuration
        :return: sampled configuration
        """
        raise NotImplementedError

    def next_moves(self, history=[], info_childs=[]):
        """Method to generate the next parameter to tune

        :param history: current incomplete configuration
        :param info_childs: information about children
        :return: tuple (next_param, value_param, is_terminal)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """Reset environment

        :param kwargs: parameter to reset environment, optional
        """
        pass

    def _evaluate(self, config):
        """Method to evaluate one configuration

        :param config: configuration to evaluate
        :return: performance of the configuration
        """
        raise NotImplementedError

    def get_nb_children(self, parameter, value, current_pipeline):
        """Get the number of

        :param parameter:
        :param value:
        :return: maximum number of children
        """
        raise NotImplementedError

    def _check_if_same_pipeline(self, pip1, pip2):
        return set(pip1) != set(pip2)

    def _has_finite_child(self, history=[]):
        try:
            rollout = self.rollout(history)
        except Exception as e:
            return False
        return self._check_if_same_pipeline([el for el in rollout], [el[0] for el in history])

    def run_default_configuration(self):
        raise NotImplementedError

    def __str__(self):
        return "Environment: %s\n\t" \
               "-> start time: %s (UTC)" % (self.__class__.__name__,
                                            datetime.datetime.utcfromtimestamp(int(self.start_time)).strftime('%Y-%m-%d %H:%M:%S'))
