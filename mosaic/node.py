import math
import networkx as nx

class Node():

    def __init__(self):
        self.tree = nx.DiGraph()
        self.id_count = -1
        self.add_node(name="root", value=None)

    def get_new_id(self):
        self.id_count += 1
        return self.id_count

    def add_node(self, name=None, value=None, visits=0, reward=0.0, terminal=False, max_number_child=1, parent_node = None):
        new_id = self.get_new_id()
        self.tree.add_node(new_id, name=name, value=value, visits=visits,
                            reward=reward, terminal=terminal,
                            max_number_child=max_number_child)
        if parent_node is not None:
            self.tree.add_path([parent_node, new_id])
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

        nb_child_allowed = min(nb_childs, self.get_attribute(node_id, "max_number_child"))
        if nb_current_childs >= nb_child_allowed:
            return True

        return False

    def backprop_from_node(self, node_id, reward):
        for parent in self.get_path_to_node(node_id=node_id, name =False):
            self.update_node(parent, reward)

    def update_node(self, node_id, reward):
        node = self.tree.nodes[node_id]
        self.tree.node[node_id]["max_number_child"] += int(math.pow(node["visits"], 0.7))
        self.tree.node[node_id]['reward'] = node["reward"] + reward
        self.tree.node[node_id]["visits"] += 1

    def get_childs(self, node_id):
        return list(self.tree.successors(node_id))

    def get_info_node(self, node_id):
        node = self.tree.nodes[node_id]
        return {"id": node_id,
                "name": node["name"],
                "value": node["value"],
                "visits": node["visits"],
                "reward": node["reward"],
                "max_number_child": node["max_number_child"]}

    def get_attribute(self, node_id, attribute):
        return self.tree.nodes[node_id][attribute]

    def set_attribute(self, node_id, attribute, new_value):
        self.tree.node[node_id][attribute] = new_value
