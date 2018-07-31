"""Space in which MCTS will be run."""


from copy import deepcopy
import random
import time

from mosaic.simulation.rules import ChildRule
from mosaic.simulation.scenario import ComplexScenario, ChoiceScenario
from mosaic.utils import random_uniform_on_log_space

class Space():
    def __init__(self, scenario = None, sampler = {}, rules = []):
        if not isinstance(scenario, ComplexScenario) and not isinstance(scenario, ChoiceScenario):
            self.scenario = ChoiceScenario(name = "root", scenarios = [scenario])
        else:
            self.scenario = scenario
        self.sampler = sampler
        self.rules = rules

    def next_params(self, history=[], info_childs = []):
        """Return next hyperparameter
            node_name: current node (string)
            history: (node_name, value)
        """
        ok = False
        while not ok:
            scenario = self.generate_playout_scenario(history = history)
            param = scenario.call()
            val = self.sample(param)
            # print(history, param, val, self.test_rules(history + [(param, val)]), info_childs)
            if (len(info_childs) == 0 or (param, val) not in info_childs) and self.test_rules(history + [(param, val)]):
                ok = True
        return param, val, self.get_nb_possible_child(history + [(param, val)]) == 0

    def can_be_executed(self, param, history):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and name in rule.applied_to:
                for parent, value_parent in history:
                    if parent == rule.parent and value_parent in rule.value:
                        return True

    def has_finite_child(self, history=[]):
        """Return True if number of child is finite
        """
        nb_child = self.get_nb_possible_child(history)

        return (nb_child == float("inf")), nb_child

    def sample(self, node_name):
        """Sample the next configuration.
        """
        random.seed(time.time())
        if node_name in self.sampler:
            parameter = self.sampler[node_name]
            return parameter.sample_new_value()
        else:
            return None

    def playout(self, history=[]):
        if self.get_nb_possible_child(history=history) == 0:
            return history

        res_final = deepcopy(history)
        final = False

        while not final:
            param, value, final = self.next_params(history = res_final)
            res_final.append((param, value))

        return res_final

    def test_rules(self, list_nodes):
        for r in self.rules:
            if not r.test(list_nodes):
                return False
        return True

    def get_possible_value(self, node_name):
        b, f = self.sampler[node_name].get_info()
        return b, f

    def get_rules(self, node_name):
        list_rules = []
        for r in self.rules:
            if node_name in r.applied_to:
                list_rules.append(r)
        return list_rules

    def get_nb_possible_child(self, history):
        nb = 0

        scenario = self.generate_playout_scenario(history = history)

        for child in scenario.queue_tasks():
            if child in self.sampler:
                value_list, type_sampling = self.get_possible_value(child)
                if type_sampling == "choice":
                    for child_value in value_list:
                        if self.test_rules(history + [(child, child_value)]):
                            nb += 1
                elif type_sampling == "constant":
                    if self.test_rules(history + [(child, value_list)]):
                        nb += 1
                elif type_sampling in ["uniform", "log_uniform"]:
                    for i in range(10):
                        child_value = self.sample(child)
                        if self.test_rules(history + [(child, child_value)]):
                            nb += 1
            else:
                nb += 1
        return nb

    def generate_playout_scenario(self, history = []):
        scenario = deepcopy(self.scenario)

        for config, value in history:
            scenario.execute(config)
            scenario.actualize_queue(config, value)

        return scenario


class Parameter():
    def __init__(self, name = None, value_list = [], type_sampling = None,
                 type = None):
        self.name = name
        self.value_list = value_list
        self.type_sampling = type_sampling
        self.type = type

        if type_sampling not in ["uniform", "choice", "constant", "log_uniform"]:
            raise Exception("Can not handle {0} type".format(self.type))

    def get_info(self):
        return self.value_list, self.type_sampling

    def sample_new_value(self):
        if self.type_sampling == "choice":
            return random.choice(self.value_list)
        elif self.type_sampling == "uniform":
            if self.type == 'int':
                return random.randint(self.value_list[0], self.value_list[1])
            else:
                return random.uniform(self.value_list[0], self.value_list[1])
        elif self.type_sampling == "constant":
            return self.value_list
        elif self.type_sampling == "log_uniform":
            return random_uniform_on_log_space(self.value_list[0], self.value_list[1])
