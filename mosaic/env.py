"""Base environement class."""

import time
import pynisher
import random

from mosaic.space import Space


class ConfigSpace_env():
    """Base class for environement."""

    terminal_state = []
    bestconfig = {
        "score": 0,
        "model": None
    }

    def __init__(self, eval_func, config_space, logfile = "", mem_in_mb=4048, cpu_time_in_s=300, num_processes=1):
        """Constructor."""
        self.history = {}
        self.start_time = time.time()
        self.logfile = logfile
        self.config_space = config_space
        self.preprocess = False

        # Constrained evaluation
        self.eval_func = pynisher.enforce_limits(mem_in_mb=mem_in_mb,
                                                 cpu_time_in_s=cpu_time_in_s,
                                                 num_processes=num_processes,
                                                 logger=None)(eval_func)

    def rollout(self, history = []):
        config = self.config_space.sample_partial_configuration(history)
        return config

    def next_moves(self, history = [], info_childs = []):
        config = self.config_space.sample_partial_configuration(history)
        #print(config)
        moves_executed = set([el[0] for el in history])
        possible_moves = set(config.keys())
        #print("Next move: {0}".format(random.sample(possible_moves - moves_executed, 1)))
        next_param = random.sample(possible_moves - moves_executed, 1)[0]
        value_param = config[next_param]
        history.append((next_param, value_param))
        return next_param, value_param, not self.has_finite_child(history)


    def _preprocess_moves_util(self, list_moves, index):
        model = list_moves[index][0]
        param_model = {}
        last_index = index
        for i in range(index + 1, len(list_moves)):
            last_index = i
            config, val = list_moves[i]
            if config.startswith(model):
                param_model[config.split("__")[1]] = val
            else:
                break
        return model, param_model, last_index

    def _preprocess_moves(self, list_moves):
        index = 0
        preprocessed_moves = []
        while (index < len(list_moves) - 1):
            model, params, index = self._preprocess_moves_util(list_moves, index)
            preprocessed_moves.append((model, params))
        return preprocessed_moves

    def _evaluate(self, list_moves):
        if self.preprocess:
            config = self._preprocess_moves(list_moves)
        else:
            config = list_moves

        hash_moves = hash(str(config))
        if hash_moves in self.history:
            return self.history[hash_moves]

        res = self.eval_func(config, self.bestconfig)

        if res > self.bestconfig["score"]:
            self.bestconfig = {
                "score": res,
                "model": config
            }
            self.log_result()

        # Add into history
        self.history[hash_moves] = res
        return res

    def has_finite_child(self, history=[]):
        rollout = self.rollout(history)
        return (len(set([el[0] for el in rollout]) - set([el[0] for el in history])) == 0)

    def log_result(self):
        if self.logfile != "":
            with open(self.logfile, "a+") as f:
                f.write("{0},{1}\n".format(time.time() - self.start_time, self.bestconfig["score"]))


class Env():
    """Base class for environement."""

    terminal_state = []
    bestconfig = {
        "score": 0,
        "model": None
    }

    def __init__(self, eval_func, scenario = None, sampler = {}, rules = [],
                 logfile = "", mem_in_mb=4048, cpu_time_in_s=300, num_processes=1):
        """Constructor."""
        self.space = Space(scenario = scenario, sampler = sampler, rules = rules)
        self.history = {}
        self.start_time = time.time()
        self.logfile = logfile
        self.preprocess = True

        # Constrained evaluation
        self.eval_func = pynisher.enforce_limits(mem_in_mb=mem_in_mb,
                                                 cpu_time_in_s=cpu_time_in_s,
                                                 num_processes=num_processes,
                                                 logger=None)(eval_func)

    def rollout(self, history = []):
        return self.space.playout(history)

    def next_moves(self, history=[], info_childs=[]):
        return self.space.next_params(history, info_childs)

    def _preprocess_moves(self, list_moves, index):
        model = list_moves[index][0]
        param_model = {}
        last_index = index
        for i in range(index + 1, len(list_moves)):
            last_index = i
            config, val = list_moves[i]
            if config.startswith(model):
                param_model[config.split("__")[1]] = val
            else:
                break
        return  model, param_model, last_index

    def preprocess_moves(self, list_moves):
        index = 0
        preprocessed_moves = []
        while(index < len(list_moves) - 1):
            model, params, index = self._preprocess_moves(list_moves, index)
            preprocessed_moves.append((model, params))
        return preprocessed_moves

    def _evaluate(self, list_moves = []):
        if self.preprocess:
            config = self.preprocess_moves(list_moves)
        else:
            config = list_moves

        hash_moves = hash(pformat(config))
        if hash_moves in self.history:
            return self.history[hash_moves]

        res = self.eval_func(config, self.bestconfig)

        if res > self.bestconfig["score"]:
            self.bestconfig = {
                "score": res,
                "model": config
            }
            self.log_result()

        # Add into history
        self.history[hash_moves] = res
        return res

    def has_finite_child(self, history=[]):
        return self.space.has_finite_child(history)

    def log_result(self):
        if self.logfile != "":
            with open(self.logfile, "a+") as f:
                f.write("{0},{1}\n".format(time.time() - self.start_time, self.bestconfig["score"]))
