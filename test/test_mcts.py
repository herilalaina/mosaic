import unittest
import random


from mosaic.env import Env
from mosaic.mcts import MCTS
from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario
from mosaic.space import Space
from mosaic.rules import ChildRule


class TestMCTS(unittest.TestCase):

    def init_mcts(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0
        x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1__p1", "x1__p2"])
        x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2__p1", "x2__p2"])

        start = ChoiceScenario(name = "root", scenarios=[x1, x2])
        sampler = { "x1__p1": ([0, 1], "uniform", "float"),
                    "x1__p2": ([[1, 2, ]], "choice", "int"),
                    "x2__p1": ([["a", "b", "c"]], "choice", "string"),
                    "x2__p2": ([[a_func, b_func, c_func]], "choice", "func"),
        }
        env = Env(scenario = start, sampler = sampler)
        def evaluate(config):
            return random.uniform(0, 1)
        Env.evaluate = evaluate
        return MCTS(env = env)

    def test_tree_policy(self):
        mcts = self.init_mcts()
        for i in range(1, 5):
            assert(mcts.TREEPOLICY() == min(3, i))

    def test_random_policy(self):
        mcts = self.init_mcts()
        node = mcts.TREEPOLICY()
        r = mcts.random_policy(node)
        assert((r >= 0) and (r <= 1))

    def test_search(self):
        mcts = self.init_mcts()
        n1 = mcts.TREEPOLICY()
        assert(n1 == 1)
        r1 = mcts.random_policy(n1)
        mcts.BACKUP(n1, r1)
        assert(mcts.tree.get_attribute(0, "reward") == r1)
        assert(mcts.tree.get_attribute(1, "reward") == r1)
        assert(len(mcts.tree.tree) == 2)

        n2 = mcts.TREEPOLICY()
        assert(n2 == 2)
        r2 = mcts.random_policy(n2)
        mcts.BACKUP(n2, r2)
        assert(mcts.tree.get_attribute(0, "reward") == r1 + r2)
        assert(mcts.tree.get_attribute(1, "reward") == r1 + r2)
        assert(mcts.tree.get_attribute(2, "reward") == r2)
        assert(len(mcts.tree.tree) == 3)

        n3 = mcts.TREEPOLICY()
        assert(n3 == 3)
        r3 = mcts.random_policy(n3)
        mcts.BACKUP(n3, r3)
        assert(mcts.tree.get_attribute(0, "reward") == r1 + r2 + r3)
        assert(mcts.tree.get_attribute(1, "reward") == r1 + r2 + r3)
        assert(mcts.tree.get_attribute(2, "reward") == r2 + r3)
        assert(mcts.tree.get_attribute(3, "reward") == r3)
        assert(len(mcts.tree.tree) == 4)

    def test_everything_with_rules(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0
        x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1__p1", "x1__p2"])
        x2 = ListTask(is_ordered=False, name = "x2",  tasks = ["x2__p1", "x2__p2"])

        start = ChoiceScenario(name = "root", scenarios=[x1, x2])
        sampler = { "x1__p1": ([0, 1], "uniform", "float"),
                    "x1__p2": ([[1, 2, ]], "choice", "int"),
                    "x2__p1": ([["a", "b", "c"]], "choice", "string"),
                    "x2__p2": ([[a_func, b_func, c_func]], "choice", "func"),
        }
        rules = [ChildRule(applied_to = ["x2__p2"], parent = "x2__p1", value = ["a", "c"])]
        env = Env(scenario = start, sampler = sampler, rules = rules)
        def evaluate(config):
            return random.uniform(0, 1)
        Env.evaluate = evaluate

        mcts = MCTS(env = env)
        mcts.run(n = 100, generate_image_path = "images")
