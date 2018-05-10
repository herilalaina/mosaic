import math
import unittest

from mosaic.node import Node
from mosaic.policy import UCT

class TestPolicy(unittest.TestCase):

    def test_uct(self):
        SCALAR = 1 / math.sqrt(2.0)
        uct_policy = UCT()
        node = Node()
        node.add_node(name="root")

        node.add_node(name="c1", parent_node = 0)
        node.backprop_from_node(1, 0.5)
        node.add_node(name="c2", parent_node = 0)
        node.backprop_from_node(2, 0.3)
        childs = [node.get_info_node(n) for n in node.get_childs(0)]
        assert(uct_policy.BESTCHILD(node.get_info_node(0), childs, SCALAR) == 1)

        node.add_node(name="b1", parent_node = 1)
        node.backprop_from_node(3, 0.1)
        node.add_node(name="b2", parent_node = 1)
        node.backprop_from_node(4, 0.75)
        childs = [node.get_info_node(n) for n in node.get_childs(1)]
        assert(uct_policy.BESTCHILD(node.get_info_node(1), childs, SCALAR) == 4)

        node.add_node(name="d1", parent_node = 3)
        node.backprop_from_node(5, 0.1)
        childs = [node.get_info_node(n) for n in node.get_childs(0)]
        assert(uct_policy.BESTCHILD(node.get_info_node(0), childs, SCALAR) == 2)
