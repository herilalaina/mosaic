import unittest

from mosaic.env import Env
from mosaic.simulation.scenario import WorkflowListTask, WorkflowChoiceScenario
from mosaic.space import Space
from mosaic.simulation.parameter import Parameter


class TestEnv(unittest.TestCase):

    def test_preprocess_moves(self):
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
        ex_config = space.playout(history=[("Model", None)])

        env = Env(a_func, scenario = start, sampler = sampler)

        for i in range(10):
            configs = env.preprocess_moves(ex_config)
            for model, param in configs:
                assert(model in ["Model", "x1", "x2"])
                for p, v in param.items():
                    if model == "x1":
                        if p == "p1":
                            assert((v >= 0) and (v <= 1))
                        else:
                            assert(v in [1, 2, 3, 4, 5, 6, 7])
                    else:
                        if p == "p1":
                            assert(v in ["a", "b", "c", "d"])
                        else:
                            assert(v in [a_func, b_func, c_func])
