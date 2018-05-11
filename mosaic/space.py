"""Space in which MCTS will be run."""


from copy import deepcopy
import random
import time

from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario

class Space():
    def __init__(self, scenario = None, sampler = {}):
        self.scenario = scenario
        self.sampler = sampler

    def sampling_method(self, is_finite):
        if is_finite == "choice":
            return random.choice
        else:
            return random.uniform

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
            if len(info_childs) == 0 or (p, v) not in info_childs:
                ok = True
        return p, v, (len(scenario.queue_tasks()) == 0)

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
                    nb_child += len(b[0])
                else:
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
