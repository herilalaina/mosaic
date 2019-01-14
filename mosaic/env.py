"""Base environement class."""

import time
import pynisher
import numpy as np

from mosaic.model_score import ScoreModel
from mosaic.utils import Timeout, get_index_percentile

from ConfigSpace.hyperparameters import CategoricalHyperparameter


class ConfigSpace_env():
    """Base class for environement."""

    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=30,
                 use_parameter_importance=True,
                 use_rave=False,
                 seed = 1):
        """Constructor."""
        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()
        self.config_space = config_space
        self.use_parameter_importance = use_parameter_importance
        self.use_rave = use_rave

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.cpu_time_in_s = cpu_time_in_s
        self.mem_in_mb = mem_in_mb
        self.eval_func = eval_func

        self.score_model = ScoreModel(len(self.config_space._hyperparameters))
        self.history_score = []
        self.final_model = []

        # statistics
        self.sucess_run = 0
        self.rng = np.random.RandomState(seed)

    def reset(self, eval_func,
              mem_in_mb=3024,
              cpu_time_in_s=30):
        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.cpu_time_in_s = cpu_time_in_s
        self.mem_in_mb = mem_in_mb
        self.eval_func = eval_func

        self.history_score = []


    def rollout(self, history=[]):
        try:
            config = self.config_space.sample_partial_configuration_with_default(history)
        except Exception as e:
            print(history)
            raise e
            #config = self.config_space.sample_partial_configuration(history)
        return config

    def next_moves(self, history=[], info_childs=[]):
        try:
            #while True:
            config = self.config_space.sample_partial_configuration_with_default(history)
            #    if self._valid_sample(history, config):
            #        break

            possible_params = list(self.config_space.get_possible_next_params(history))
            print(possible_params)
            if self.use_parameter_importance:
                id_param = self.score_model.most_importance_parameter(
                    [self.config_space.get_idx_by_hyperparameter_name(p) for p in possible_params])
            else:
                id_param = np.random.randint(0, len(possible_params))
            next_param = possible_params[id_param]

            next_param_cs = self.config_space._hyperparameters[next_param]
            value_to_choose = []

            for _ in range(100):
                next_param_v = next_param_cs.sample(self.rng)
                try:
                    self.config_space.sample_partial_configuration_with_default(history + [(next_param, next_param_v)])
                    value_to_choose.append(next_param_v)
                except Exception as e:
                    pass



            if self.use_rave:
                idx_param = self.config_space.get_idx_by_hyperparameter_name(next_param)
                if type(self.config_space._hyperparameters[next_param]) == CategoricalHyperparameter:
                    value_param = self.score_model.rave_value(
                        [next_param_cs._inverse_transform(v) for v in value_to_choose],
                        idx_param,
                        True,
                        self.config_space._hyperparameters[next_param].choices)
                    value_param = next_param_cs._transform(value_param)
                else:
                    value_param = self.score_model.rave_value(value_to_choose,
                                                              idx_param,
                                                              False,
                                                              None)
            else:
                value_param = np.random.choice(value_to_choose)

            history.append((next_param, value_param))
        except Timeout.Timeout as e:
            raise (e)
        except Exception as e:
            print("Exception for {0}".format(history))
            raise (e)
        is_terminal = not self._check_if_same_pipeline([el[0] for el in history], [el for el in config])
        return next_param, value_param, is_terminal

    def _valid_sample(self, history, config):
        for hp_name, hp_value in history:
            if hp_name not in config.keys() or config[hp_name] != hp_value:
                return False
        return True

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

    def _evaluate(self, config, default=False):
        if not default:
            eval_func = pynisher.enforce_limits(mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.cpu_time_in_s)(self.eval_func)
        else:
            eval_func = self.eval_func
        try:
            res = eval_func(config, self.bestconfig)
            self.sucess_run += 1
        except Timeout.Timeout as e:
            print("Timeout!!")
            raise (e)
        except Exception as e:
            print("Pynisher Error {0}. Config: {1}".format(e, config))
            res = {"validation_score": 0, "model": None}
            raise e

        if res is None:
            res = {"validation_score": 0, "model": None}

        if res["validation_score"] > 0:
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), res["validation_score"])

        self.log_result(res, config)

        """self.log_result(res, config)

        percentile = 1 - (1 / (self.sucess_run + 1))
        index = get_index_percentile([r["validation_score"] for r in self.history_score], percentile)
        if self.history_score[index] != self.bestconfig["model"]:
            self.add_to_final_model(res, self.history_score[index])
            self.bestconfig = self.history_score[index]
            print("!! {0}: validation score: {1}\n".format(str(config), res["validation_score"]))
        else:
            print(">> {0}: validation score: {1}\n".format(str(config), res["validation_score"]))"""

        return res["validation_score"]

    def run_default_configuration(self):
        try:
            self._evaluate(self.config_space.get_default_configuration(), default=True)
        except:
            pass

    def run_random_configuration(self):
        try:
            self._evaluate(self.config_space.sample_configuration())
        except:
            pass

    def _check_if_same_pipeline(self, pip1, pip2):
        return set(pip1) != set(pip2)

    def has_finite_child(self, history=[]):
        rollout = self.rollout(history)
        return self._check_if_same_pipeline([el for el in rollout], [el[0] for el in history])

    def add_to_final_model(self, config):
        self.final_model.append(config)

    def log_result(self, res, config):
        run = {
            "running_time": time.time() - self.start_time,
            "validation_score": res["validation_score"],
            "test_score": res["test_score"] if "test_score" in res else None,
            "model": config
        }
        self.history_score.append(run)

        print(">> {0}: validation score: {1}\n".format(str(config), res["validation_score"]))

        if res["validation_score"] > self.bestconfig["validation_score"]:
            self.add_to_final_model(run)
            self.bestconfig = run
