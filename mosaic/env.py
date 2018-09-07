"""Base environement class."""

import time

from pprint import pformat

from mosaic.space import Space


class Env():
    """Base class for environement."""

    terminal_state = []
    bestconfig = {
        "score": 0,
        "model": None
    }

    def __init__(self, scenario = None, sampler = {}, rules = [], logfile = ""):
        """Constructor."""
        self.space = Space(scenario = scenario, sampler = sampler, rules = rules)
        self.history = {}
        self.start_time = time.time()
        self.logfile = logfile
        self.preprocess = True

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

        res = Env.evaluate(config, self.bestconfig)

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

    @staticmethod
    def evaluate(config, bestconfig):
        """Method for moves evaluation."""
        return 0
