"""Base environement class."""

import time
import pynisher
import numpy as np
import pickle as pkl

from mosaic.utils import expected_improvement, probability_improvement
from mosaic.model_score import ScoreModel
from mosaic.utils import Timeout, get_index_percentile

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, IntegerHyperparameter

import traceback
import logging



class ConfigSpace_env():
    """Base class for environement."""

    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=30,
                 use_parameter_importance=True,
                 multi_objective=False,
                 seed = 1):
        """Constructor."""
        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()
        self.config_space = config_space
        self.use_parameter_importance = use_parameter_importance
        self.multi_objective = multi_objective

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.cpu_time_in_s = cpu_time_in_s
        self.mem_in_mb = mem_in_mb
        self.eval_func = eval_func

        self.score_model = ScoreModel(len(self.config_space._hyperparameters),
                                        id_most_import_class=[self.config_space.get_idx_by_hyperparameter_name(p) for p in ["categorical_encoding:__choice__", "classifier:__choice__"]])
        self.history_score = []
        self.logger = logging.getLogger('mcts')
        self.final_model = []

        # statistics
        self.sucess_run = 0
        self.rng = np.random.RandomState(seed)

        self.id = 0
        self.main_hyperparameter = ["classifier:__choice__", "preprocessor:__choice__", "categorical_encoding"]
        self.max_nb_child_main_parameter = {
            "root": 16,
            "classifier:__choice__:adaboost": 10,
            "classifier:__choice__:bernoulli_nb": 13,
            "classifier:__choice__:decision_tree": 10,
            "classifier:__choice__:extra_trees": 10,
            "classifier:__choice__:gaussian_nb": 10,
            "classifier:__choice__:gradient_boosting": 10,
            "classifier:__choice__:k_nearest_neighbors": 10,
            "classifier:__choice__:lda": 12,
            "classifier:__choice__:liblinear_svc": 13,
            "classifier:__choice__:libsvm_svc": 10,
            "classifier:__choice__:multinomial_nb": 8,
            "classifier:__choice__:passive_aggressive": 13,
            "classifier:__choice__:qda": 12,
            "classifier:__choice__:random_forest": 10,
            "classifier:__choice__:sgd": 13,
            "classifier:__choice__:xgradient_boosting": 10,
            "preprocessor:__choice__": 2
        }
        self.nb_choice_hyp = self._get_nb_choice_for_each_parameter()

        self.nb_exec_for_params = dict()
        for p in self.config_space.get_hyperparameter_names():
            self.nb_exec_for_params[p] = {"nb": 0, "ens": set()}

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

    def _has_multiple_value(self, p):
        next_param_cs = self.config_space._hyperparameters[p]
        ens_val = set()
        for _ in range(5):
            ens_val.add(next_param_cs.sample(self.rng))
        return len(ens_val) > 1

    def _can_use_parameter_importance(self, list_params, threshold = 10):
        for p in list_params:
            if self.nb_exec_for_params[p]["nb"] < 10 or (self._has_multiple_value(p) and len(self.nb_exec_for_params[p]["ens"]) == 1):
                return False
        return True

    def importance_surgate(self, param, history):
        list_configuration_to_choose = []
        value_to_choose = set()
        next_param_cs = self.config_space._hyperparameters[param]
        for _ in range(1000):
            next_param_v = next_param_cs.sample(self.rng)
            try:
                ex_config = self.config_space.sample_partial_configuration(history + [(param, next_param_v)])
                vect_config = np.nan_to_num(ex_config.get_array())
                if len(vect_config) == 172 and next_param_v not in value_to_choose:
                    list_configuration_to_choose.append(vect_config)
                    value_to_choose.append(next_param_v)
            except Exception as e:
                pass
        mu, sigma = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model)
        return np.mean(mu)

    def can_be_selectioned(self, possible_params, child_info, history):
        final_available_param = []
        for p in possible_params:
            buffer = set()
            p_cs = self.config_space.get_hyperparameter(p)
            for _ in range(100):
                try:
                    new_val = p_cs.sample(self.rng)
                    self.config_space.sample_partial_configuration_with_default(history + [(p, new_val)])
                    buffer.add(new_val)
                except Exception as e:
                    pass

            if len([param for param in child_info if param == p]) < len(buffer):
                final_available_param.append(p)
        return final_available_param


    def next_moves(self, history=[], info_childs=[]):
        try:
            config = self.config_space.sample_partial_configuration_with_default(history)

            possible_params_ = list(self.config_space.get_possible_next_params(history))
            possible_params = self.can_be_selectioned(possible_params_, [v[0] for v in info_childs], history)

            if set(self.main_hyperparameter).intersection(set(possible_params)):
                for main_p in self.main_hyperparameter:
                    if main_p in possible_params:
                        id_param = possible_params.index(main_p)
                        break
            elif self.use_parameter_importance and self._can_use_parameter_importance(possible_params):
                print("Parameter importance activated")
                id_param = self.score_model.most_importance_parameter([self.config_space.get_idx_by_hyperparameter_name(p) for p in possible_params])
            else:
                id_param = np.random.randint(0, len(possible_params))
            next_param = possible_params[id_param]

            next_param_cs = self.config_space._hyperparameters[next_param]

            value_to_choose = []
            list_configuration_to_choose = []

            for _ in range(100):
                next_param_v = next_param_cs.sample(self.rng)
                if next_param_v not in [v[1] for v in info_childs if v[0] == next_param]:
                    try:
                        ex_config = self.config_space.sample_partial_configuration_with_default(history + [(next_param, next_param_v)])
                        vect_config = np.nan_to_num(ex_config.get_array())
                        if next_param_v not in value_to_choose:
                            value_to_choose.append(next_param_v)
                            list_configuration_to_choose.append(vect_config)
                    except Exception as e:
                        print("Error", next_param_v)

            if not self.multi_objective:
                mu, sigma = None, None
                try:
                    mu, sigma = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model)
                    ei_values = expected_improvement(mu, sigma, self.bestconfig["validation_score"])
                    value_param = value_to_choose[np.argmax(ei_values)]
                except Exception as e:
                    value_param = np.random.choice(value_to_choose)
            else:
                try:
                    mu_perf, sigma_perf = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model)
                    mu_time, sigma_time = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model_of_time)
                    ei_values_perf = probability_improvement(mu_perf, sigma_perf, self.bestconfig["validation_score"])
                    ei_values_time = probability_improvement(mu_time, sigma_time, self.bestconfig["running_time"], greater_is_better=False)
                    #print(next_param)
                    #print(value_to_choose)
                    #print("Best perf", self.bestconfig["validation_score"])
                    #print("mu perf", mu_perf)
                    #print("sigma perf", sigma_perf)
                    #print("Best time", self.bestconfig["running_time"])
                    #print("mu time", mu_time)
                    #print("sigma time", sigma_time)
                    #print(ei_values_perf)
                    #print(ei_values_time)
                    #tau = (time.time() - self.start_time) / 3600.0
                    #ei_values = [((1 - tau) * (ei_t)) + (tau * ei_p) for ei_t, ei_p in zip(ei_values_time, ei_values_perf)]
                    ei_values = [ei if mu_t < self.cpu_time_in_s else -1000 for ei, mu_t in zip(ei_values_perf, mu_time)]
                    if sum(ei_values) == -1000 * len(ei_values):
                        ei_values = ei_values_perf
                    value_param = value_to_choose[np.argmax(ei_values)]
                    #print("tau", tau)
                    #print("choosed", np.argmax(ei_values))
                except Exception as e:
                    print("error in ei")
                    value_param = np.random.choice(value_to_choose)


            history.append((next_param, value_param))
        except Timeout.Timeout as e:
            raise (e)
        except Exception as e:
            print("Exception for {0}".format(history))
            print("Child info", info_childs)
            print("Possible params", possible_params)
            print("Next params", next_param)
            print("Value to choose", value_to_choose)
            print("Value example: ", next_param_v)
            raise (e)

        print("Possible params", possible_params)
        print("Next params", (next_param, value_param))
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
            eval_func = pynisher.enforce_limits(mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.max_eval_time)(self.eval_func)
        else:
            eval_func = pynisher.enforce_limits(mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.max_eval_time)(self.eval_func)
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
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), res["validation_score"], res["running_time"])
        else:
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), -1, 3000)

        self.log_result(res, config)

        return res["validation_score"]

    def run_default_configuration(self):
        try:
            self._evaluate(self.config_space.get_default_configuration(), default=True)
        except Exception as e:
            raise(e)

    def run_main_configuration(self):
        for cl in ["bernoulli_nb", "multinomial_nb", "decision_tree", "gaussian_nb", "sgd", "passive_aggressive", "xgradient_boosting", "adaboost", "extra_trees", "gradient_boosting", "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors"]:
            try:
                config = self.config_space.sample_partial_configuration_with_default([("classifier:__choice__", cl)])
                self._evaluate(config)
            except Exception as e:
                print("Error when executing ", cl)

    def run_random_configuration(self):
        try:
            self._evaluate(self.config_space.sample_configuration())
        except:
            pass

    def _get_nb_choice_for_each_parameter(self):
        count_dict = {}
        for hyp in self.config_space.get_hyperparameter_names():
            hyp_cs = self.config_space.get_hyperparameter(hyp)
            if isinstance(hyp_cs, CategoricalHyperparameter):
                count_dict[hyp] = len(hyp_cs.choices)
            elif isinstance(hyp_cs, IntegerHyperparameter):
                buffer = set()
                for _ in range(100):
                    buffer.add(hyp_cs.sample(self.rng))
                count_dict[hyp] = len(buffer)
            else:
                count_dict[hyp] = 100
        return count_dict

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
        run["model"] = config.get_dictionary()
        for k, v in config.get_dictionary().items():
            self.nb_exec_for_params[k]["nb"] = self.nb_exec_for_params[k]["nb"] + 1
            self.nb_exec_for_params[k]["ens"].add(v)

        self.history_score.append(run)
        self.id += 1

        print(">> {0}: validation score: {1}\n".format(str(config), res["validation_score"]))

        if res["validation_score"] > self.bestconfig["validation_score"]:
            self.add_to_final_model(run)
            self.bestconfig = run
