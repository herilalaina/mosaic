"""Monte carlo tree seach class."""

import logging
import os

import numpy as np
import time

from mosaic.node import Node
from mosaic.strategy.policy import UCT, Besa, PUCT
from mosaic.utils import Timeout


class MCTS():
    """Monte carlo tree search implementation."""

    def __init__(self, env,
                 policy="uct",
                 time_budget=3600,
                 policy_arg=None,
                 exec_dir=""):
        self.env = env
        self.time_budget = time_budget
        self.exec_dir = exec_dir
        self.bestconfig = None
        self.bestscore = - np.inf

        # Init tree
        self.tree = Node()

        # Set up logger
        self.logger = logging.getLogger('mcts')

        # Policy
        if policy == "uct":
            if "c_ucb" in policy_arg:
                c_ucb = policy_arg["c_ucb"]
            else:
                c_ucb = np.sqrt(2)
            self.policy = UCT(c_ucb)
        elif policy == "besa":
            self.policy = Besa()
        elif policy == "puct":
            policy_arg["start_time"] = self.env.start_time
            policy_arg["time_budget"] = self.time_budget
            self.policy = PUCT(self.env, self.tree, policy_arg)
        else:
            raise NotImplemented("Policy {0} not implemented".format(policy))

        # iteration logging
        self.n_iter = 0

        if "proba" in policy_arg:
            self.env.proba_expert = policy_arg["proba"]

        if "coef_progressive_widening" in policy_arg:
            self.tree.coef_progressive_widening = policy_arg["coef_progressive_widening"]

    def reset(self, time_budget=3600):
        self.time_budget = time_budget
        self.n_iter = 0

    def MCT_SEARCH(self):
        """Monte carlo tree search iteration."""
        self.logger.info(
            "#########################Iteration={0}##################################".format(self.n_iter))
        self.logger.info("Begin SELECTION")
        front = self.TREEPOLICY()
        self.logger.info("End SELECTION")

        self.logger.info("Begin PLAYOUT")
        reward, config = self.PLAYOUT(front)
        self.logger.info("End PLAYOUT")

        if config is None:
            return 0, None

        self.logger.info("Begin BACKUP")
        self.BACKUP(front, reward)
        self.logger.info("End BACKUP")
        self.n_iter += 1

        return reward, config

    def TREEPOLICY(self):
        """Selection using policy."""
        node = 0  # Root of the tree
        while not self.tree.is_terminal(node):
            if len(self.tree.get_children(node)) == 0:
                return self.EXPAND(node)
            else:
                if not self.tree.fully_expanded(node, self.env):
                    self.logger.info("Not fully expanded.")
                    return self.EXPAND(node)
                else:
                    current_node = self.tree.get_info_node(node)
                    children = [[n,
                                 self.tree.get_attribute(n, "reward"),
                                 self.tree.get_attribute(n, "visits")] for n in self.tree.get_children(node) if
                                not self.tree.get_attribute(n, "invalid")]
                    if len(children) > 0:
                        node = self.policy.selection((current_node["reward"], current_node["visits"]),
                                                     [x[0] for x in children],
                                                     [x[1] for x in children],
                                                     [x[2] for x in children],
                                                     state=self.tree.get_path_to_node(node))
                        self.logger.info("Selection\t node={0}".format(node))
                    else:
                        self.logger.error(
                            "Empty list of valid children\n current node {0}\t List of children {1}".format(
                                current_node,
                                self.tree.get_children(node)))
                        return node
        return node

    def EXPAND(self, node):
        """Expand child node."""
        st_time = time.time()
        self.logger.info("Expand on node {0}\n Current history: {1}".format(node, self.tree.get_path_to_node(node)))
        name, value, terminal = self.policy.expansion(self.env.next_move,
                                                      [self.tree.get_path_to_node(node),
                                                       self.tree.get_children(node, info=["name", "value"])])
        id = self.tree.add_node(name=name, value=value,
                                terminal=terminal, parent_node=node)
        self.logger.info("Expand\t id={0}\t name={1}\t value={2}\t terminal={3}".format(
            id, name, value, terminal))
        return id

    def PLAYOUT(self, node_id):
        """Playout policy."""

        self.logger.info("Playout on : {0}".format(self.tree.get_path_to_node(node_id)))

        st_time = time.time()
        try:
            playout_node = self.env.rollout(self.tree.get_path_to_node(node_id))
        except Exception as e:
            self.logger.error("Add node %s to not possible state: %s" % (node_id, e))
            self.tree.set_attribute(node_id, "invalid", True)
            return 0, None

        score = self.policy.evaluate(self.env._evaluate, [playout_node])

        self.logger.info(
            "Playout\t param={0}\t score={1}\t exec time={2}".format(playout_node, score, time.time() - st_time))
        return score, playout_node

    def BACKUP(self, node, reward):
        """Back propagate reward."""
        for parent in self.tree.get_path_to_node(node_id=node, name=False):
            vl, vs = self.tree.get_attribute(
                parent, "reward"), self.tree.get_attribute(parent, "visits")
            new_val, new_vis = self.policy.backpropagate(
                parent, vl, vs, reward)
            self.tree.set_attribute(parent, "reward", new_val)
            self.tree.set_attribute(parent, "visits", new_vis)

    def run(self, n=1, initial_configurations=[], nb_iter_to_generate_img=-1):
        start_run = time.time()
        with Timeout(int(self.time_budget - (start_run - time.time()))):
            try:
                self.logger.info("Run default configuration")
                self.env.run_default_configuration()

                if len(initial_configurations) > 0:
                    self.logger.info("Run initial configurations")
                else:
                    self.logger.info("No initial configuration to run.")

                for i in range(n):
                    if time.time() - self.env.start_time < self.time_budget:
                        res, config = self.MCT_SEARCH()

                        if res > self.bestscore:
                            self.bestscore = res
                            self.bestconfig = config
                    else:
                        return 0

                    if nb_iter_to_generate_img == -1 or i % nb_iter_to_generate_img == 0:
                        self.tree.draw_tree(
                            os.path.join(self.exec_dir, "images"))
                        self.print_tree("tree_{0}".format(i))

            except Timeout.Timeout:
                self.logger.info("Budget exhausted.")
                return 0

    def print_tree(self, name_img):
        img_dir = os.path.join(self.exec_dir, "images")
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        self.tree.draw_tree(os.path.join(img_dir, name_img))
