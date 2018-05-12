"""Space in which MCTS will be run."""


from copy import deepcopy
import random
import time

from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario

class Space():
    def __init__(self, scenario = None, sampler = {}, rules = []):
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
            if (len(info_childs) == 0 or (param, val) not in info_childs) and self.test_rules(history + [(param, val)]):
                ok = True
        return param, val, self.get_nb_possible_child(history + [(param, val)]) == 0

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
        scenario = self.generate_playout_scenario(history = history)

        while(not scenario.finished()):
            param = scenario.call()
            value_param = self.sample(param)
            history.append((param, value_param))

        return history

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
                elif type_sampling == "uniform":
                    return float('inf')
            else:
                nb += 1
        return nb

    def generate_playout_scenario(self, history = []):
        scenario = deepcopy(self.scenario)
        for config, value in history:
            scenario.execute(config)
        return scenario


class Parameter():
    def __init__(self, name = None, value_list = [], type_sampling = None,
                 type = None):
        self.name = name
        self.value_list = value_list
        self.type_sampling = type_sampling
        self.type = type

        if type_sampling not in ["uniform", "choice", "constant"]:
            raise Exception("Can not handle {0} type".format(self.type))

    def get_info(self):
        return self.value_list, self.type_sampling

    def sample_new_value(self):
        if self.type_sampling == "choice":
            return random.choice(self.value_list)
        elif self.type_sampling == "uniform":
            return random.uniform(self.value_list[0], self.value_list[1])
        else:
            return self.value_list


"""
def a_func(): return 0
def b_func(): return 0
def c_func(): return 0

x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1_p1", "x1_p2"])
x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2_p1", "x2_p2"])

start = ChoiceScenario(name = "Model", scenarios=[x1, x2])

sampler = { "x1_p1": ([0, 1], "uniform", "float"),
            "x1_p2": ([[1, 2, 3, 4, 5, 6, 7]], "choice", "int"),
            "x2_p1": ([["a", "b", "c", "d"]], "choice", "string"),
            "x2_p2": ([[a_func, b_func, c_func]], "choice", "func"),
}

space = Space(scenario = start, sampler = sampler)

space.playout(history=[("Model", None)])
"""
