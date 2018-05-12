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

    def sampling_method(self, is_finite):
        if is_finite == "choice":
            return random.choice
        else:
            return random.uniform

    def is_valid(self, config):
        for n, v in config:
            exec("{0}={1}".format(n, v))
        for r in self.rules:
            if not eval(r):
                return False
        return True

    def next_params(self, history=[], info_childs = []):
        """Return next hyperparameter
            node_name: current node (string)
            history: (node_name, value)
        """
        ok = False
        while not ok:
            scenario = deepcopy(self.scenario)
            for config, _ in history:
                scenario.execute(config)
            p = scenario.call()
            v = self.sample(p)
            if (len(info_childs) == 0 or (p, v) not in info_childs) and self.test_rules(history + [(p, v)]):
                ok = True
        return p, v, self.get_nb_possible_child(scenario, history + [(p, v)]) == 0

    def has_finite_child(self, history=[]):
        """Return True if number of child is finite
        """
        scenario = deepcopy(self.scenario)
        for config, _ in history:
            scenario.execute(config)
        nb_child = 0
        for next_task in scenario.queue_tasks():
            if next_task in self.sampler:
                b, f, t = self.sampler[next_task]

                if f == "choice":
                    for child_value in b[0]:
                        if self.test_rules(history + [(next_task, child_value)]):
                            nb_child += 1
                else:
                    if self.test_rules(history + [(next_task, None)]):
                        return True, 99999
            else:
                nb_child += 1
        return False, nb_child

    def sample(self, node_name):
        """Sample the next configuration.
        """
        random.seed(time.time())
        if node_name in self.sampler:
            b, f, t = self.sampler[node_name]
            method = self.sampling_method(f)
            return method(*b)
        else:
            return None

    def playout(self, history=[]):
        scenario = deepcopy(self.scenario)

        for config, value in history:
            scenario.execute(config)

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
        b, f, t = self.sampler[node_name]
        return b, f

    def get_rules(self, node_name):
        list_rules = []
        for r in self.rules:
            if node_name in r.applied_to:
                list_rules.append(r)
        return list_rules

    def get_nb_possible_child(self, scenario, history):
        nb = 0
        for child in scenario.queue_tasks():
            b, f = self.get_possible_value(child)
            if f == "choice":
                for child_value in b[0]:
                    if self.test_rules(history + [(child, child_value)]):
                        nb += 1
            else:
                if self.test_rules(history + [(child, None)]):
                    nb += 1
        return nb


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
