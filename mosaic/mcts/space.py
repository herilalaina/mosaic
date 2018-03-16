import random

class Space():

    def __init__(self):
        self.sampler = dict()


    def sample(self, node_name, moves=[]):
        child = self.get_child(node_name)
        if len(child) == 0:
            return None, None
        elif len(child) == 1:
            name = list(child.keys())[0]
            func, params = self.sampler[name]
            return name, func(*params)
        elif len(child) >= 2: # Two childs
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
        return Space._get_child(self.root, node_name)


    def set_sampler(self, sampler):
        self.sampler = sampler


class Node_space():
    def __init__(self, name, parent = {}):
        self.name = name
        self.children = {}
        self.parent = parent

    def add_child(self, child_name):
        child = Node_space(child_name, parent={self.name: self})
        self.children[child_name] = child
        return child

    def append_child(self, child_node):
        self.children[child_node.name] = child_node
        child_node.parent[self.name] = self

    def append_childs(self, child_nodes):
        for child_node in child_nodes:
            self.append_child(child_node)
