"""Monte carlo tree seach class."""

import random

import logging
import math
import networkx as nx

from mosaic.policy import UCT
from mosaic.space import Space
from mosaic.node import Node


class MCTS():
    """Monte carlo tree search implementation."""

    def __init__(self, env, logfile=''):
        """Initialization."""
        # environement
        self.env = env

        # Init tree
        self.root = nx.DiGraph()
        n = Node("root", value=None)
        root_node = Node(name="root", value=None)
        self.root.add_node("root", name="root", att=n)

        # Set up logger
        if logfile != '':
            self.logger = logging.getLogger('mcts')
            hdlr = logging.FileHandler(logfile, mode='w')
            formatter = logging.Formatter('%(asctime)s %(message)s')
            hdlr.setFormatter(formatter)
            self.logger.addHandler(hdlr)
            self.logger.setLevel(logging.DEBUG)

        # Policy
        self.policy = UCT()

    def MCT_SEARCH(self):
        """MCTS method."""
        front = self.TREEPOLICY()
        reward = self.random_policy(front)
        self.BACKUP(front, reward)
        #return self.policy.BESTCHILD(, 0)

    def TREEPOLICY(self):
        """Search for the best child node."""
        SCALAR = 1 / math.sqrt(2.0)

        node = "root"

        while not nx.get_node_attributes(self.root, "att")[node].terminal:
            childs = list(self.root.successors(node))
            if len(childs) == 0:
                return self.EXPAND(node)
            else:
                if not self.root.nodes[node]["att"].fully_expanded(self.env.space, self.root):
                    return self.EXPAND(node)
                else:
                    node = self.policy.BESTCHILD(self.root.nodes[node]["att"], self.list_node(node), SCALAR)
            if len(childs) == 0:
                break

        return node

    def list_node(self, node):
        return [v["att"] for k, v in self.root.nodes.items()]

    def EXPAND(self, node):
        """Expand child node."""
        tried_children = set(self.root.successors(node))
        next_node = self.env.space.next_params(node, history = [v.name for k, v in self.root.predecessors(node).items()])
        while True:
            name, val = self.env.space.sample(next_node)
            final_name = name + "=" + str(val)
            if final_name not in tried_children:
                break
        node_to_add = Node(name=final_name, value=val)
        self.root.add_node(final_name, name=val, att=node_to_add)
        self.root.add_path([node, final_name])
        return final_name

    def random_policy(self, node):
        """Random policy."""
        playout_node = self.env.space.playout(self.root, node)
        return self.env._evaluate(self.root)

    def BACKUP(self, node, reward):
        """Back propagate reward."""
        for parent in list(nx.ancestors(self.root, node)):
            self.root.nodes[parent]["att"].update(reward)

    def run(self, n=1):
        """Play 1 simulation."""
        for i in range(n):
            self.MCT_SEARCH()
            print(self.env.bestscore)
        #return self.env.best_model
