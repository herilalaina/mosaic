import unittest

from mosaic.simulation.scenario import WorkflowListTask, WorkflowComplexScenario, WorkflowChoiceScenario
class TestScenario(unittest.TestCase):

    def test_call(self):
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

    def test_execute(self):
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

    def test_queue_task(self):
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

    def test_finished(self):
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

    def test_choice_scenario(self):
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

    def test_choice_complex_scenario(self):
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
