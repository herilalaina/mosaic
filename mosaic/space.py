"""Space in which MCTS will be run."""


from copy import deepcopy
import random
import time

from mosaic.simulation.scenario import WorkflowComplexScenario, WorkflowChoiceScenario, AbstractImportanceScenario


class Space():
    def __init__(self, scenario = None, sampler = {}, rules = []):
        if not isinstance(scenario, WorkflowComplexScenario) and not isinstance(scenario, WorkflowChoiceScenario)\
                and not isinstance(scenario, AbstractImportanceScenario):
            self.scenario = WorkflowChoiceScenario(name ="root", scenarios = [scenario])
        else:
            self.scenario = scenario
        self.sampler = sampler
        self.rules = rules

    def next_params(self, history=[], info_children = []):
        """Return next hyperparameter
            node_name: current node (string)
            history: (node_name, value)
        """
        ok = False
        while not ok:
            scenario = self.generate_playout_scenario(history = history)
            param = scenario.call()
            val = self.sample(param)
            if (len(info_children) == 0 or (param, val) not in info_children) and self.test_rules(history + [(param, val)]):
                ok = True
        return param, val, self.get_nb_possible_child(history + [(param, val)]) == 0

    def has_finite_child(self, history=[]):
        """Return True if number of child is finite"""
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