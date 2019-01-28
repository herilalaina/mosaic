"""Base environement class."""

import time
import pynisher
import numpy as np

from mosaic.utils import expected_improvement
from mosaic.model_score import ScoreModel
from mosaic.utils import Timeout, get_index_percentile

import traceback
import logging



class ConfigSpace_env():
    """Base class for environement."""

    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=30,
                 use_parameter_importance=True,
                 seed = 1):
        """Constructor."""
        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()
        self.config_space = config_space
        self.use_parameter_importance = use_parameter_importance

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.cpu_time_in_s = cpu_time_in_s
        self.mem_in_mb = mem_in_mb
        self.eval_func = eval_func

        self.score_model = ScoreModel(len(self.config_space._hyperparameters))
        self.history_score = []
        self.logger = logging.getLogger('mcts')
        self.final_model = []

        # statistics
        self.sucess_run = 0
        self.rng = np.random.RandomState(seed)

        self.id = 0

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
        return config

    def next_moves(self, history=[], info_childs=[]):
        try:
            config = self.config_space.sample_partial_configuration_with_default(history)

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
            list_configuration_to_choose = []

            for _ in range(100):
                next_param_v = next_param_cs.sample(self.rng)
                try:
                    ex_config = self.config_space.sample_partial_configuration_with_default(history + [(next_param, next_param_v)])
                    vect_config = np.nan_to_num(ex_config.get_array())
                    if len(vect_config) == 172 and next_param_v not in value_to_choose:
                        value_to_choose.append(next_param_v)
                        list_configuration_to_choose.append(vect_config)
                except Exception as e:
                    pass

            if len(self.history_score) > 10:
                mu, sigma = None, None
                try:
                    mu, sigma = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose))
                    ei_values = expected_improvement(mu, sigma, self.bestconfig["validation_score"])
                    value_param = value_to_choose[np.argmax(ei_values)]
                except Exception as e:
                    print("ERROR =============================================")
                    print(e)
                    print(mu)
                    print(sigma)
                    print(ei_values)
                    print(value_to_choose)
                    print(np.shape(list_configuration_to_choose))
                    value_param = np.random.choice(value_to_choose)
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
        start_time = time.time()
        if not default:
            eval_func = pynisher.enforce_limits(mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.cpu_time_in_s)(self.eval_func)
        else:
            eval_func = self.eval_func
        try:
            res = eval_func(config, self.bestconfig)
            self.sucess_run += 1

        except TimeoutException as e:
            self.logger.critical(e)
            raise(e)

        if res is None:
            res = {"validation_score": 0, "info": None}

        res["running_time"] = time.time() - start_time
        res["predict_performance"] = self.score_model.get_performance(np.nan_to_num(config.get_array()))

        if res["validation_score"] > 0:
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), res["validation_score"])

        self.log_result(res, config)

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
        run = res
        run["id"] = self.id
        run["elapsed_time"] = time.time() - self.start_time
        #run["model"] = config.get_dictionary()

        self.history_score.append(run)
        self.id += 1

        print(">> {0}: validation score: {1}\n".format(str(config), res["validation_score"]))

        if res["validation_score"] > self.bestconfig["validation_score"]:
            self.add_to_final_model(run)
            self.bestconfig = run
