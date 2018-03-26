"""Space in which MCTS will be run."""

import random


class Space():
    """Abstract space class."""

    def __init__(self):
        """Initialization."""
        self.root = Node_space("root")
        self.sampler = dict()
        self.terminal_pointer = []

    def sample(self, node_name, moves=[]):
        """Sample the next configuration."""
        child = self.get_child(node_name)
        if len(child) == 0:
            return None, None
        elif len(child) == 1:
            name = list(child.keys())[0]
            func, params = self.sampler[name]
            return name, func(*params)
        elif len(child) >= 2:  # Two childs
            arr_child = [c for c in child]
            v = random.choice(arr_child)
            return v, v
        else:
            raise Exception('Error handling this case.')

    def _get_child(node, node_name):
        if node.name == node_name:
            return node.children
        else:
            for c in node.children.keys():
                res = Space._get_child(node.children[c], node_name)
                if isinstance(res, dict) and res != {}:
                    return res
        return {}

    def get_child(self, node_name):
        """Get child of node."""
        return Space._get_child(self.root, node_name)

    def set_sampler(self, sampler):
        """Set sampler."""
        self.sampler = sampler


class Node_space():
    """Node space class."""

    def __init__(self, name, parent={}):
        """Initialization."""
        self.name = name
        self.children = {}
        self.parent = parent

    def add_child(self, child_name):
        """Add child to space."""
        child = Node_space(child_name, parent={self.name: self})
        self.children[child_name] = child
        return child

    def append_child(self, child_node):
        """Append node as a child."""
        self.children[child_node.name] = child_node
        child_node.parent[self.name] = self

    def append_childs(self, child_nodes):
        """Append a list of child."""
        for child_node in child_nodes:
            self.append_child(child_node)
