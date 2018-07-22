"""Monte carlo tree seach class."""

import logging
import math

from mosaic.strategy.uct import UCT
from mosaic.node import Node


class MCTS():
    """Monte carlo tree search implementation."""

    def __init__(self, env, policy=UCT()):
        self.env = env

        # Init tree
        self.tree = Node()

        # Set up logger
        self.logger = logging.getLogger('mcts')

        # Policy
        self.policy = policy

    def MCT_SEARCH(self):
        """Monte carlo tree search iteration."""
        front = self.TREEPOLICY()
        reward = self.PLAYOUT(front)
        self.BACKUP(front, reward)

    def TREEPOLICY(self):
        """Selection using policy."""

        node = 0 # Root of the tree
        while not self.tree.is_terminal(node):
            if len(self.tree.get_childs(node)) == 0:
                return self.EXPAND(node)
            else:
                if not self.tree.fully_expanded(node, self.env.space):
                    return self.EXPAND(node)
                else:
                    current_node = self.tree.get_info_node(node)
                    children = [[self.tree.nodes(n)["reward"],
                                 self.tree.nodes(n)["visits"]] for n in self.tree.get_childs(node)]
                    node = self.policy.selection((current_node["reward"], current_node["visits"]),
                                                 children[:, 0],
                                                 children[:, 1])
        return node

    def EXPAND(self, node):
        """Expand child node."""
        name, value, terminal = self.env.space.next_params(self.tree.get_path_to_node(node),
                                                            self.tree.get_childs(node, info = ["name", "value"]))
        return self.tree.add_node(name=name, value=value, terminal=terminal, parent_node = node)

    def PLAYOUT(self, node_id):
        """Playout policy."""
        playout_node = self.env.rollout(self.tree.get_path_to_node(node_id))
        score = self.env._evaluate(playout_node)
        return score

    def BACKUP(self, node, reward):
        """Back propagate reward."""
        for parent in self.tree.get_path_to_node(node_id=node, name=False):
            vl, vs = self.tree.get_attribute(parent, "reward"), self.tree.get_attribute(parent, "visits")
            new_val, new_vis = self.policy.backpropagate(vl, vs, reward)
            self.tree.set_attribute(parent, "reward", new_val)
            self.tree.set_attribute(parent, "visits", new_vis)

    def run(self, n = 1, generate_image_path = ""):
        for i in range(n):
            self.MCT_SEARCH()
            if generate_image_path != "":
                self.tree.draw_tree("{0}/{1}.png".format(generate_image_path, i))
            print(".", end = "")
        print("")
