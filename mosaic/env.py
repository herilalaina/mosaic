"""Base environement class."""

import time
import datetime
import numpy as np


class AbstractEnvironment:
    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=300,
                 seed=42):
        """Abstract class for environment

        :param eval_func: evaluation function
        :param config_space: configuration space
        :param mem_in_mb: memory budget
        :param cpu_time_in_s: time budget for each run
        :param seed: random seed
        """
        self.eval_func = eval_func
        self.config_space = config_space
        self.mem_in_mb = mem_in_mb
        self.cpu_time_in_s = cpu_time_in_s
        self.seed = seed
        self.start_time = time.time()
        self.rng = np.random.RandomState(seed)

    def rollout(self, history=[]):
        """Rollout method to generate complete configuration starting with `history`

        :param history: current incomplete configuration
        :return: sampled configuration
        """
        raise NotImplemented

    def next_moves(self, history=[], info_childs=[]):
        """Method to generate the next parameter to tune

        :param history: current incomplete configuration
        :param info_childs: information about children
        :return: tuple (next_param, value_param, is_terminal)
        """
        raise NotImplemented

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
        raise NotImplemented

    def get_nb_childs(self, parameter, value, current_pipeline):
        """Get the number of

        :param parameter:
        :param value:
        :return: maximum number of children
        """
        raise NotImplemented

    def _check_if_same_pipeline(self, pip1, pip2):
        return set(pip1) != set(pip2)

    def _has_finite_child(self, history=[]):
        rollout = self.rollout(history)
        return self._check_if_same_pipeline([el for el in rollout], [el[0] for el in history])

    def __str__(self):
        return "Environment: %s\n\t" \
               "-> evaluation function: %s \n\t" \
               "-> memory limit: %s Mb\n\t" \
               "-> cpu time limit for each run: %s sec\n\t" \
               "-> start time: %s (UTC)" % (self.__class__.__name__, self.eval_func, self.mem_in_mb, self.cpu_time_in_s,
                                            datetime.utcfromtimestamp(int(self.start_time)).strftime('%Y-%m-%d %H:%M:%S'))
