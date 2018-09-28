"""Base environement class."""

import time
import pynisher
from pprint import pformat

from mosaic.space import Space


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

    def log_result(self):
        if self.logfile != "":
            with open(self.logfile, "a+") as f:
                f.write("{0},{1}\n".format(time.time() - self.start_time, self.bestconfig["score"]))
