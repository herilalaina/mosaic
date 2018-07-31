import unittest

from mosaic.space import Space
from mosaic.simulation.parameter import Parameter
from mosaic.simulation.rules import ChildRule
from mosaic.simulation.scenario import WorkflowListTask, WorkflowChoiceScenario

class TestSpace(unittest.TestCase):

    def test_init(self):
        pass

    def test_next_params(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0

        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2", "x2__p3"])

        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "func"),
                    "x2__p3": Parameter("x2__p3", "lol", "constant", "string"),
        }

        space = Space(scenario = start, sampler = sampler)
        for i in range(50):
            assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "c"), ("x2__p2", a_func)]) == ("x2__p3", "lol", True))
            assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "c")])[0] == "x2__p2")
            assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[0] == "x2__p2")
            assert(space.next_params(history=[("Model", None), ("x2", None)])[0] == "x2__p1")
            assert(space.next_params(history=[("Model", None), ("x1", None)])[0] in ["x1__p1", "x1__p2"])
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p2", 5)])[0] in ["x1__p1", "x1__p2"])
            assert(space.next_params(history=[("Model", None)])[0] in ["x1", "x2"])

    def test_is_valid(self):
        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2", "x2__p3"])

        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [10, 11, 12], "choice", "int"),
                    "x2__p3": Parameter("x2__p3", "lol", "constant", "string"),
        }
        rules = [ChildRule(applied_to = ["x2__p2"], parent = "x2__p1", value = ["a"])]


        space = Space(scenario = start, sampler = sampler, rules = rules)
        assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[0] in ["x2__p2", "x2__p3"])
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "b")])[1] == 0)
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[1] > 0)


    def test_sample(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0

        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2"])

        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "func"),
        }

        space = Space(scenario = start, sampler = sampler)

        for i in range(10):
            v = space.sample("x1__p1")
            assert(v >= 0)
            assert(v <= 1)
            assert(space.sample("x1__p2") in [1, 2, 3, 4, 5, 6, 7])
            assert(space.sample("x2__p1") in ["a", "b", "c", "d"])
            assert(space.sample("x2__p2") in [a_func, b_func, c_func])

    def test_playout(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0

        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2"])

        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "func"),
        }

        space = Space(scenario = start, sampler = sampler)

        for i in range(10):
            space.playout(history = [("Model", None)])

    def test_has_finite_child(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0

        x1 = WorkflowListTask(is_ordered=False, name ="x1", tasks = ["x1__p1", "x1__p2"])
        x2 = WorkflowListTask(is_ordered=True, name ="x2", tasks = ["x2__p1", "x2__p2"])

        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [a_func, b_func, c_func], "choice", "func"),
        }

        space = Space(scenario = start, sampler = sampler)

        assert(space.has_finite_child(history = [("Model", None)]) == (False, 2))
        assert(space.has_finite_child(history = [("Model", None), ("x1", None)]) == (False, 17))
        assert(space.has_finite_child(history = [("Model", None), ("x1", None), ("x1__p1", 0.5)]) == (False, 7))
        assert(space.has_finite_child(history = [("Model", None), ("x1", None), ("x1__p1", 0.5), ("x1__p2", 1)]) == (False, 0))
        assert(space.has_finite_child(history = [("Model", None), ("x2", None)]) == (False, 4))
        assert(space.has_finite_child(history = [("Model", None), ("x2", None), ("x2__p1", "a")]) == (False, 3))
        assert(space.has_finite_child(history = [("Model", None), ("x2", None), ("x2__p1", "c"), ("x2__p2", c_func)]) == (False, 0))
