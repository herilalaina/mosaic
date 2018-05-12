import unittest

from mosaic.rules import ChildRule, ValueRule
from mosaic.space import Space, Parameter
from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario

class TestRules(unittest.TestCase):

    def test_child_rules(self):
        x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1__p1", "x1__p2"])
        x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2__p1", "x2__p2"])

        start = ChoiceScenario(name = "Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [10, 11, 12], "choice", "int")
        }

        rules = [ChildRule(applied_to = ["x2__p2"], parent = "x2__p1", value = ["a"])]


        space = Space(scenario = start, sampler = sampler, rules = rules)
        assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[0] == "x2__p2")
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "b")])[1] == 0)
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[1] > 0)


    def test_value_rules(self):
        x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1__p1", "x1__p2"])
        x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2__p1", "x2__p2"])

        start = ChoiceScenario(name = "Model", scenarios=[x1, x2])

        sampler = { "x1__p1": Parameter("x1__p1", [0, 1], "uniform", "float"),
                    "x1__p2": Parameter("x1__p2", [1, 2, 3, 4, 5, 6, 7], "choice", "int"),
                    "x2__p1": Parameter("x2__p1", ["a", "b", "c", "d"], "choice", "string"),
                    "x2__p2": Parameter("x2__p2", [10, 11, 12], "choice", "int")
        }

        rules = [ChildRule(applied_to = ["x2__p2"], parent = "x2__p1", value = ["a"]),
                 ValueRule(constraints = [("x1__p1", 0.5), ("x1__p2", 7)]),
                 ValueRule(constraints = [("x1__p1", 0.9), ("x1__p2", 6)])]

        space = Space(scenario = start, sampler = sampler, rules = rules)
        assert(space.next_params(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[0] == "x2__p2")
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "b")])[1] == 0)
        assert(space.has_finite_child(history=[("Model", None), ("x2", None), ("x2__p1", "a")])[1] > 0)

        for i in range(10):
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p1", 0.3)]) != ("x1__p2", 7, True))
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p1", 0.5)]) == ("x1__p2", 7, True))
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p1", 0.8)]) != ("x1__p2", 7, True))
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p1", 0.9)]) != ("x1__p2", 7, True))
            assert(space.next_params(history=[("Model", None), ("x1", None), ("x1__p1", 0.9)]) == ("x1__p2", 6, True))
