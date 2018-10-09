"""Monte carlo tree seach class."""

import logging
import math

from mosaic.strategy.policy import UCT, Besa
from mosaic.node import Node


class MCTS():
    """Monte carlo tree search implementation."""

    def __init__(self, env, policy="uct", knowledge=None):
        self.env = env

        # Init tree
        self.tree = Node()

        # Set up logger
        self.logger = logging.getLogger('mcts')

        # Policy
        if policy == "uct":
            self.policy = UCT()
        elif policy == "besa":
            self.policy = Besa()
        else:
            raise NotImplemented("Policy {0} not implemented".format(policy))

        # Knowledge
        self.knowledge = knowledge

        # iteration logging
        self.n_iter = 0

    def MCT_SEARCH(self):
        """Monte carlo tree search iteration."""
        self.logger.info("#########################Iteration={0}##################################".format(self.n_iter))
        front = self.TREEPOLICY()
        reward = self.PLAYOUT(front)
        self.BACKUP(front, reward)

        self.n_iter += 1

    def TREEPOLICY(self):
        """Selection using policy."""
        node = 0 # Root of the tree
        while not self.tree.is_terminal(node):
            if len(self.tree.get_childs(node)) == 0:
                return self.EXPAND(node)
            else:
                if not self.tree.fully_expanded(node, self.env):
                    return self.EXPAND(node)
                else:
                    current_node = self.tree.get_info_node(node)
                    children = [[n,
                                 self.tree.get_attribute(n, "reward"),
                                 self.tree.get_attribute(n, "visits")] for n in self.tree.get_childs(node)]
                    node = self.policy.selection((current_node["reward"], current_node["visits"]),
                                                 [x[0] for x in children],
                                                 [x[1] for x in children],
                                                 [x[2] for x in children])
                    self.logger.info("Selection\t node={0}".format(node))
        return node

    def EXPAND(self, node):
        """Expand child node."""
        name, value, terminal = self.policy.expansion(self.env.next_moves,
                                                      [self.tree.get_path_to_node(node),
                                                       self.tree.get_childs(node, info = ["name", "value"])])
        id = self.tree.add_node(name=name, value=value, terminal=terminal, parent_node = node)
        self.logger.info("Expand\t id={0}\t name={1}\t value={2}\t terminal={3}".format(id, name, value, terminal))
        return id

    def PLAYOUT(self, node_id):
        """Playout policy."""
        playout_node = self.env.rollout(self.tree.get_path_to_node(node_id))
        score = self.policy.evaluate(self.env._evaluate, [playout_node])
        self.logger.info("Playout\t param={0}\t score={1}".format(playout_node, score))
        return score

    def BACKUP(self, node, reward):
        """Back propagate reward."""
        for parent in self.tree.get_path_to_node(node_id=node, name=False):
            vl, vs = self.tree.get_attribute(parent, "reward"), self.tree.get_attribute(parent, "visits")
            new_val, new_vis = self.policy.backpropagate(parent, vl, vs, reward)
            self.tree.set_attribute(parent, "reward", new_val)
            self.tree.set_attribute(parent, "visits", new_vis)

    def run(self, n = 1, generate_image_path = ""):
        for i in range(n):
            self.MCT_SEARCH()
            if generate_image_path != "":
                self.tree.draw_tree("{0}/{1}.png".format(generate_image_path, i))
            print(" ", end = "")
        print("")
