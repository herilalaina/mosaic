import unittest
import random


from mosaic.env import Env
from mosaic.mcts import MCTS
from mosaic.simulation.scenario import WorkflowListTask, WorkflowChoiceScenario
from mosaic.simulation.parameter import Parameter
from mosaic.simulation.rules import ChildRule, ValueRule


class TestMCTS(unittest.TestCase):

    def init_mcts(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0
        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2"])

        start = WorkflowChoiceScenario(name ="root", scenarios=[x1, x2])
        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "int")
        }

        env = Env(a_func, scenario = start, sampler = sampler)
        def evaluate(config, bestconfig):
            return random.uniform(0, 1)
        Env.evaluate = evaluate
        return MCTS(env = env)

    def test_everything_with_rules(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0
        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2", "x1__p3"])
        x2 = WorkflowListTask(is_ordered=False, name ="x2", tasks = ["x2__p1", "x2__p2", "x2__p3"])

        start = WorkflowChoiceScenario(name ="root", scenarios=[x1, x2])
        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2], "choice", "int"),
                    "x1__p3": Parameter("x1__p3", ["v", "w", "x", "y", "z"], "choice", "string"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "int"),
                    "x2__p3": Parameter("x2__p3", "lol", "constant", "string"),
        }
        rules = [
            ChildRule(applied_to = ["x2__p2"], parent = "x2__p1", value = ["a", "c"]),
            ValueRule(constraints = [("x1__p2", 1), ("x1__p3", "v")]),
            ValueRule(constraints = [("x1__p2", 2), ("x1__p3", "w")])
        ]
        def evaluate(config, bestconfig):
            return random.uniform(0, 1)
        env = Env(evaluate, scenario = start, sampler = sampler, rules = rules)

        mcts = MCTS(env = env)
        mcts.run(n = 100)