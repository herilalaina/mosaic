"""Base environement class."""

import datetime
import logging

import numpy as np
import time


class AbstractEnvironment:
    def __init__(self, seed=42):
        """Abstract class for environment

        :param seed: random seed
        """
        self.seed = seed
        self.logger = logging.getLogger("mcts")
        self.start_time = time.time()
        self.rng = np.random.RandomState(seed)

    def init_configurations(self, intial_configurations):
        for config in intial_configurations:
            self.evaluate(config)

    def _evaluate(self, config):
        self.logger.info("Evaluation configuration: {0}".format(config))
        score = self.evaluate(config)
        self.logger.info("Score: {0}".format(score))
        return score

    def _check_if_same_pipeline(self, pip1, pip2):
        return set(pip1) != set(pip2)

    def _has_finite_nb_children(self, history=[]):
        try:
            rollout = self.rollout(history)
        except Exception as e:
            return False
        return self._check_if_same_pipeline([el for el in rollout], [el[0] for el in history])

    def run_default_configuration(self):
        raise NotImplementedError

    def serialize_configuration(self, config):
        return config

    def __str__(self):
        return "Environment: %s\n\t" \
               "-> start time: %s (UTC)" % (self.__class__.__name__,
                                            datetime.datetime.utcfromtimestamp(int(self.start_time)).strftime(
                                                '%Y-%m-%d %H:%M:%S'))


class MosaicEnvironment(AbstractEnvironment):
    def __init__(self, seed=42):
        """Init method for environment

        :param seed: environment seed
        :type seed: int
        """
        super().__init__(seed)

    def rollout(self, history=[]):
        """Rollout method to generate complete configuration starting with `history`

        :param history: current incomplete configuration
        :type history: List of (name_param:str, value_param:object)
        :return: sampled configuration: list of tuple (name_param:str, value_param:object)
        """
        raise NotImplementedError

    def next_move(self, history=[], info_children=[]):
        """Method to generate the next parameter to tune

        :param history: current incomplete configuration
        :param info_children: information about children
        :return: tuple (next_param:str, value_param:obbejct, is_terminal:bool)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """Reset environment.

        :param kwargs: method to reset environment, optional
        """
        pass

    def evaluate(self, config):
        """Method to evaluate one configuration

        :param config: configuration to evaluate. List of tuple (name_param:str, value_param:object)
        :return: performance of the configuration: Float
        """
        raise NotImplementedError

    def get_nb_children(self, parameter, value, current_pipeline):
        """Get the number of

        :param parameter:name
        :param value:value
        :return: maximum number of children:int
        """
        raise NotImplementedError
