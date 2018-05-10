import unittest

from mosaic.node import Node
from mosaic.space import Space
from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario

class TestNode(unittest.TestCase):

    def test_add_node(self):
        node = Node()
        assert(node.id_count == 0)
        assert(node.get_attribute(0, "name") == "root")
        node.add_node(name="v1", parent_node = 0)
        assert(node.get_path_to_node(1, name=False) == [0, 1])

    def test_get_path_to_node(self):
        node = Node()
        node.add_node(name="c1", parent_node = 0)
        node.add_node(name="c2", parent_node = 0)
        node.add_node(name="b1", parent_node = 1)
        node.add_node(name="b2", parent_node = 1)
        node.add_node(name="d1", parent_node = 3)
        assert(node.get_path_to_node(1, name=False) == [0, 1])
        assert(node.get_path_to_node(2, name=False) == [0, 2])
        assert(node.get_path_to_node(3, name=False) == [0, 1, 3])
        assert(node.get_path_to_node(4, name=False) == [0, 1, 4])
        assert(node.get_path_to_node(5, name=False) == [0, 1, 3, 5])

    def test_backprop_from_node(self):
        node = Node()

        node.add_node(name="c1", parent_node = 0)
        node.backprop_from_node(1, 0.5)
        assert(node.get_attribute(0, "visits") == node.get_attribute(1, "visits"))
        assert(node.get_attribute(0, "reward") == node.get_attribute(1, "reward"))

        node.add_node(name="c2", parent_node = 0)
        node.backprop_from_node(2, 0.3)
        assert(node.get_attribute(0, "visits") == 2)
        assert(node.get_attribute(0, "reward") == 0.8)

        node.add_node(name="b1", parent_node = 1)
        node.backprop_from_node(3, 0.1)
        assert(node.get_attribute(0, "visits") == 3)
        assert(node.get_attribute(1, "visits") == 2)
        assert(node.get_attribute(0, "reward") == 0.9)
        assert(node.get_attribute(1, "reward") == 0.6)

        node.add_node(name="b2", parent_node = 1)
        node.backprop_from_node(4, 0.75)
        assert(node.get_attribute(0, "visits") == 4)
        assert(node.get_attribute(1, "visits") == 3)
        assert(node.get_attribute(0, "reward") == 1.65)
        assert(node.get_attribute(1, "reward") == 1.35)

        node.add_node(name="d1", parent_node = 3)
        node.backprop_from_node(5, 0.5)
        assert(node.get_attribute(0, "visits") == 5)
        assert(node.get_attribute(1, "visits") == 4)
        assert(node.get_attribute(3, "visits") == 2)
        assert(node.get_attribute(0, "reward") == 2.15)
        assert(node.get_attribute(1, "reward") == 1.85)
        assert(node.get_attribute(3, "reward") == 0.6)

    def test_fully_expanded(self):
        def a_func(): return 0
        def b_func(): return 0
        def c_func(): return 0

        x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1__p1", "x1__p2"])
        x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2__p1", "x2__p2"])

        start = ChoiceScenario(name = "root", scenarios=[x1, x2])

        sampler = { "x1__p1": ([0, 1], "uniform", "float"),
                    "x1__p2": ([[1, 2]], "choice", "int"),
                    "x2__p1": ([["a", "b", "c", "d"]], "choice", "string"),
                    "x2__p2": ([[a_func, b_func, c_func]], "choice", "func"),
        }
        space = Space(scenario = start, sampler = sampler)

        node = Node()
        node.add_node(name="x1", parent_node = 0)
        node.add_node(name="x2", parent_node = 0)
        assert(node.fully_expanded(0, space))
        assert(not node.fully_expanded(1, space))
        assert(not node.fully_expanded(2, space))

        node.add_node(name="x1__p1", parent_node = 1)
        assert(node.fully_expanded(1, space))
        node.set_attribute(1, "max_number_child", 2)
        assert(not node.fully_expanded(1, space))
        assert(not node.fully_expanded(3, space))
        node.set_attribute(3, "max_number_child", 2)

        node.add_node(name="x1__p2", value=1, parent_node = 3)
        assert(not node.fully_expanded(3, space))
        node.add_node(name="x1__p2", value=2, parent_node = 3)
        assert(node.fully_expanded(3, space))
