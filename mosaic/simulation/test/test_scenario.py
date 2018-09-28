import unittest

from mosaic.simulation import scenario
from mosaic.simulation.scenario import *

class TestScenario(unittest.TestCase):

    def test_constructor(self):
        importance_scenario = scenario.ImportanceScenarioStatic({}, [])
        assert(isinstance(importance_scenario, scenario.AbstractImportanceScenario))
        assert (isinstance(importance_scenario, scenario.BaseScenario))

        for class_scenario in [scenario.WorkflowListTask, scenario.WorkflowChoiceScenario, scenario.WorkflowComplexScenario]:
            workflow_scenario = class_scenario()
            assert (isinstance(workflow_scenario, scenario.AbstractWorkflowScenario))
            assert (isinstance(workflow_scenario, scenario.BaseScenario))


    def test_importance_static(self):
        graph = {
            "root": ["algo"],
            "algo": ["algo__param1"],
            "algo__param1": ["algo__param2"],
            "algo__param2": ["algo__param3"],
            "algo__param3": ["algo__param4"]
        }

        sc = scenario.ImportanceScenarioStatic(graph, [])
        assert (sc.call() == "root")
        assert(sc.call() == "algo")
        assert (sc.call() == "algo__param1")
        assert (sc.call() == "algo__param2")
        assert (sc.call() == "algo__param3")
        assert (sc.call() == "algo__param4")

        sc = scenario.ImportanceScenarioStatic(graph, [])
        for task in ["root", "algo", "algo__param1", "algo__param2", "algo__param3", "algo__param4"]:
            assert (sc.execute(task) == task)

    def test_workflow_call(self):
        arr1 = ["x1_p1", "x1_p2"]
        arr2 = ["x2_p1", "x2_p2"]

        x1 = WorkflowListTask(name ="x1", is_ordered=False, tasks = arr1.copy())
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2.copy())

        start = WorkflowComplexScenario(name ="Model", scenarios=[x1, x2], is_ordered=True)

        assert(start.call() == "Model")
        assert(start.call() == "x1")
        assert(start.call() in ["x1_p1", "x1_p2"])
        assert(start.call() in ["x1_p1", "x1_p2"])
        assert(start.call() == "x2")
        assert(start.call() == "x2_p1")
        assert(start.call() == "x2_p2")

    def test_workflow_execute(self):
        arr1 = ["x1_p1"]
        arr2 = ["x2_p1", "x2_p2"]

        x1 = WorkflowListTask(name ="x1", is_ordered=False, tasks = arr1.copy())
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2.copy())

        start = WorkflowComplexScenario(name ="Model", scenarios=[x1, x2], is_ordered=True)

        assert(start.execute("Model") == "Model")
        assert(start.execute("x1") == "x1")
        assert(start.execute("x1_p1") == "x1_p1")
        assert(start.execute("x2") == "x2")
        assert(start.execute("x2_p1") == "x2_p1")
        assert(start.execute("x2_p2") == "x2_p2")

    def test_workflow_queue_task(self):
        arr1 = ["x1_p1"]
        arr2 = ["x2_p1", "x2_p2"]

        x1 = WorkflowListTask(name ="x1", is_ordered=False, tasks = arr1.copy())
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2.copy())

        start = WorkflowComplexScenario(name ="Model", scenarios=[x1, x2], is_ordered=True)

        assert(start.queue_tasks() == ["Model"])
        start.call()
        assert(start.queue_tasks() == ["x1"])
        start.call()
        assert(start.queue_tasks() == ["x1_p1"])
        start.call()
        assert(start.queue_tasks() == ["x2"])
        start.call()
        assert(start.queue_tasks() == ["x2_p1"])

    def test_workflow_finished(self):
        arr1 = ["x1_p1", "x1_p2"]
        arr2 = ["x2_p1", "x2_p2"]
        x1 = WorkflowListTask(name ="x1", is_ordered=False, tasks = arr1)
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2)
        start = WorkflowComplexScenario(name ="Model", scenarios=[x1, x2], is_ordered=True)

        for t in range(4):
            start.call()
        assert(x1.finished())

        for t in range(3):
            start.call()
        assert(x2.finished())
        assert(start.finished())

    def test_workflow_choice_scenario(self):
        arr1 = ["x1_p1", "x1_p2"]
        arr2 = ["x2_p1", "x2_p2"]
        x1 = WorkflowListTask(name ="x1", is_ordered=False, tasks = arr1)
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2)
        start = WorkflowChoiceScenario(name ="Model", scenarios=[x1, x2])
        assert(start.call() == "Model")
        assert(start.queue_tasks() == ["x1", "x2"])
        start.execute("x2")
        assert(start.call() == "x2_p1")
        assert(start.call() == "x2_p2")
        assert(start.finished())

    def test_workflow_choice_complex_scenario(self):
        arr1 = ["x1_p1", "x1_p2"]
        arr2 = ["x2_p1", "x2_p2"]
        arr3 = ["x3_p1", "x3_p2"]
        arr4 = ["x4_p1", "x4_p2"]

        x1 = WorkflowListTask(name ="x1", is_ordered=True, tasks = arr1)
        x2 = WorkflowListTask(name ="x2", is_ordered=True, tasks = arr2)
        x3 = WorkflowListTask(name ="x3", is_ordered=True, tasks = arr3)
        x4 = WorkflowListTask(name ="x4", is_ordered=True, tasks = arr4)

        c1 = WorkflowChoiceScenario(name ="choix_1", scenarios=[x1, x2])
        c2 = WorkflowChoiceScenario(name ="choix_2", scenarios=[x3, x4])

        start = WorkflowComplexScenario(name ="Model", scenarios=[c1, c2], is_ordered = True)

        assert(start.call() == "Model")
        assert(start.queue_tasks() == ["choix_1"])
        assert(start.call() == "choix_1")
        assert(start.call() in ["x1", "x2"])
        assert(start.call() in ["x1_p1", "x2_p1"])
        assert(start.call() in ["x1_p2", "x2_p2"])
        assert(start.call() == "choix_2")
        assert(start.call() in ["x3", "x4"])
        assert(start.call() in ["x3_p1", "x4_p1"])
        assert(start.call() in ["x3_p2", "x4_p2"])
