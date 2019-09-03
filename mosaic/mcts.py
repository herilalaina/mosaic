"""Monte carlo tree seach class."""

import logging
import os
import gc
import time
import numpy as np
import json

from mosaic.strategy.policy import UCT, Besa, PUCT
from mosaic.node import Node
from mosaic.utils import Timeout
from mosaic.utils import get_index_percentile
from networkx.readwrite.gpickle import write_gpickle


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
        front = self.TREEPOLICY()
        reward, config = self.PLAYOUT(front)
        self.BACKUP(front, reward)
        self.n_iter += 1

        # if self.exec_dir != "":
        #     # self.env.score_model.save_data(self.exec_dir)
        #     #write_gpickle(self.tree, os.path.join(self.exec_dir, "tree.json"))
        #     with open(os.path.join(self.exec_dir, "full_log.json"), 'w') as outfile:
        #         json.dump(self.env.history_score, outfile)

        return reward, config

    def TREEPOLICY(self):
        """Selection using policy."""
        node = 0  # Root of the tree
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
                                                 [x[2] for x in children],
                                                 state=self.tree.get_path_to_node(node))
                    self.logger.info("Selection\t node={0}".format(node))
        return node

    def EXPAND(self, node):
        """Expand child node."""
        st_time = time.time()
        name, value, terminal = self.policy.expansion(self.env.next_moves,
                                                      [self.tree.get_path_to_node(node),
                                                       self.tree.get_childs(node, info=["name", "value"])])
        id = self.tree.add_node(name=name, value=value,
                                terminal=terminal, parent_node=node)
        #print("Expand: ", time.time() - st_time, " sec")
        self.logger.info("Expand\t id={0}\t name={1}\t value={2}\t terminal={3}".format(
            id, name, value, terminal))
        return id

    def PLAYOUT(self, node_id):
        """Playout policy."""

        st_time = time.time()
        playout_node = self.env.rollout(self.tree.get_path_to_node(node_id))

        score = self.policy.evaluate(self.env._evaluate, [playout_node])
        if score > 0:
            self.logger.info(
                "Playout\t param={0}\t score={1}\t exec time={2}".format(playout_node, score, time.time() - st_time))
            return score, playout_node

        print("Evaluate: ", time.time() - st_time, " sec")

        self.logger.info(
            "Playout\t param={0}\t score={1}".format(playout_node, 0))
        return 0, playout_node

    def BACKUP(self, node, reward):
        """Back propagate reward."""
        for parent in self.tree.get_path_to_node(node_id=node, name=False):
            vl, vs = self.tree.get_attribute(
                parent, "reward"), self.tree.get_attribute(parent, "visits")
            new_val, new_vis = self.policy.backpropagate(
                parent, vl, vs, reward)
            self.tree.set_attribute(parent, "reward", new_val)
            self.tree.set_attribute(parent, "visits", new_vis)

    def create_node_for_algorithm(self):
        id_class = {}
        for cl in ["bernoulli_nb", "multinomial_nb",
                   "decision_tree", "gaussian_nb", "sgd",
                   "passive_aggressive", "xgradient_boosting",
                   "adaboost", "extra_trees", "gradient_boosting",
                   "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors", "random_forest"]:
            id_class[cl] = self.tree.add_node(
                name="classifier:__choice__", value=cl, terminal=False, parent_node=0)
        return id_class

    def run(self, n=1, intial_configuration=[], generate_image_path=""):
        start_run = time.time()

        self.bestconfig = None
        self.bestscore = - np.inf

        with Timeout(int(self.time_budget - (start_run - time.time()))):
            try:
                self.env.run_default_configuration()

                for i in range(n):
                    if time.time() - self.env.start_time < self.time_budget:
                        res, config = self.MCT_SEARCH()

                        # if res > self.bestscore:
                        #     self.bestscore = res
                        #     self.bestconfig = config
                    else:
                        return 0
                    gc.collect()

                if self.exec_dir != "":
                    self.tree.draw_tree(
                        os.path.join(self.exec_dir, "images"))

            except Timeout.Timeout:
                return 0

    def print_tree(self, images):
        self.tree.draw_tree(images)
