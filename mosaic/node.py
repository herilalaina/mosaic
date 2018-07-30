import matplotlib.pyplot as plt
import math
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import graphviz_layout


class Node():

    def __init__(self):
        self.tree = nx.DiGraph()
        self.id_count = -1
        self.add_node(name = "root", value=None)

    def get_new_id(self):
        self.id_count += 1
        return self.id_count

    def add_node(self, name=None, value=None, visits=0, reward=0.0, terminal=False, max_number_child=1, parent_node = None):
        new_id = self.get_new_id()
        self.tree.add_node(new_id, name=name, value=value, visits=visits,
                            reward=reward, terminal=terminal,
                            max_number_child=max_number_child)
        if parent_node is not None:
            self.tree.add_edge(parent_node, new_id)
        return new_id

    def is_terminal(self, node_id):
        return self.tree.nodes[node_id]["terminal"]

    def get_path_to_node(self, node_id, name = True):
        path = nx.shortest_path(self.tree, source=0, target=node_id)
        if name:
            return [(self.tree.nodes[v]["name"], self.tree.nodes[v]["value"]) for v in path]
        return path

    def fully_expanded(self, node_id, space):
        # Check if node is fully expanded.
        is_finite, nb_childs = space.has_finite_child(self.get_path_to_node(node_id))
        nb_current_childs = len(list(self.tree.successors(node_id)))
        nb_child_allowed = min(nb_childs, math.floor(math.sqrt(self.get_attribute(node_id, "visits"))))
        if nb_current_childs >= nb_child_allowed:
            return True

        return False

    def get_label(self, node_id):
        node = self.tree.nodes[node_id]
        node_val = node["value"] if not callable(node["value"]) else node["value"].__name__
        if node["value"] is not None:
            return "{0}={1}\n({2}, {3})".format(node["name"],
                                                  str(node_val)[:7],
                                                  str(node["visits"]),
                                                  str(node["reward"])[:4])
        else:
            return "{0}\n({1}, {2})".format(node["name"],
                                                  str(node["visits"]),
                                                  str(node["reward"])[:4])

    def update_labels(self):
        for i in range(len(self.tree)):
            self.tree.node[i]["label"] = self.get_label(i)

    def draw_tree(self, file_name = ""):
        self.update_labels()
        write_dot(self.tree, file_name + '.dot')

    def get_childs(self, node_id, info = []):
        if len(info) == 0:
            return list(self.tree.successors(node_id))
        else:
            res = []
            for c in list(self.tree.successors(node_id)):
                val = (self.tree.nodes[c][info[0]], )
                for i in range(1, len(info)):
                    val = val + (self.tree.nodes[c][info[i]], )
                res.append(val)
            return res

    def get_info_node(self, node_id):
        node = self.tree.nodes[node_id]
        return {"id": node_id,
                "name": node["name"],
                "value": node["value"] if not callable(node["value"]) else node["value"].__name__,
                "visits": node["visits"],
                "reward": node["reward"],
                "max_number_child": node["max_number_child"]}

    def get_attribute(self, node_id, attribute):
        return self.tree.nodes[node_id][attribute]

    def set_attribute(self, node_id, attribute, new_value):
        self.tree.node[node_id][attribute] = new_value
