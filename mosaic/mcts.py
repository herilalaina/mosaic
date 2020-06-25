"""Monte Carlo-Tree Search Class."""

import os
import logging
import time

import numpy as np

from mosaic.node import Node
from mosaic.utils import Timeout
from mosaic.strategy.policy import UCT, Besa, PUCT


class MCTS:
    """
    Implementation of Monte Carlo Tree Search algorithm

    Parameters
    -----------
    env: object
        Problem environment
    time_budget : int
        Time budget
    coef_progressive_widening: float
        Coefficient of progressive widening
    exec_dir: str
        Path to store results

    Attributes
    ----------
    best_config: object
        Current best configuration
    best_score: float
        Current best score
    tree: object <class mosaic.node.Node>
        Tree created by the MCTS algorithm
    logger: object
        Logger
    policy: object
        Bandit algorithm used
    n_iter: int
        Number of executed MCTS simulation
        (selection, expansion, playout, back-propagation)

    """

    def __init__(self,
                 env,
                 time_budget,
                 bandit_policy,
                 coef_progressive_widening,
                 exec_dir):
        self.env = env
        self.time_budget = time_budget
        self.exec_dir = exec_dir
        self.best_config = None
        self.best_score = - np.inf

        # Init tree
        self.tree = Node()

        # Set up logger
        self.logger = logging.getLogger('mcts')

        # Policy
        if bandit_policy["policy_name"] == "uct":
            if "c_ucb" in bandit_policy:
                c_ucb = bandit_policy["c_ucb"]
            else:
                c_ucb = np.sqrt(2)
            self.policy = UCT(c_ucb)
        elif bandit_policy["policy_name"] == "besa":
            self.policy = Besa()
        elif bandit_policy["policy_name"] == "puct":
            bandit_policy["start_time"] = self.env.start_time
            bandit_policy["time_budget"] = self.time_budget
            self.policy = PUCT(self.env, self.tree, bandit_policy)
        else:
            raise NotImplemented("Policy {0} not implemented".format(bandit_policy["policy_name"]))

        self.n_iter = 0
        self.tree.coef_progressive_widening = coef_progressive_widening

    def MCT_SEARCH(self):
        """One simulation of MCTS.

        One simulation is composed of selection,
        expansion, playout and back-propagation

        Returns:
        --------
            reward: float
                Reward of the simulation
            config: object
                Configuration run

        """
        self.logger.info(
            "## ITERATION={0} ##".format(self.n_iter))
        front = self.TREEPOLICY()

        reward, config = self.PLAYOUT(front)

        if config is None:
            return 0, None

        self.BACKUP(front, reward)
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
                    # self.logger.info("Not fully expanded.")
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
                        # self.logger.info("Selection\t node={0}".format(node))
                    else:
                        # self.logger.error(
                        #     "Empty list of valid children\n current node {0}\t List of children {1}".format(
                        #         current_node,
                        #         self.tree.get_children(node)))
                        return node
        return node

    def EXPAND(self, node):
        """Expand child node."""
        st_time = time.time()
        # self.logger.info("Expand on node {0}\n Current history: {1}".format(node, self.tree.get_path_to_node(node)))
        name, value, terminal = self.policy.expansion(self.env.next_move,
                                                      [self.tree.get_path_to_node(node),
                                                       self.tree.get_children(node, info=["name", "value"])])
        id = self.tree.add_node(name=name, value=value,
                                terminal=terminal, parent_node=node)
        # self.logger.info("Expand\t id={0}\t name={1}\t value={2}\t terminal={3}".format(
            # id, name, value, terminal))
        return id

    def PLAYOUT(self, node_id):
        """Playout policy."""

        # self.logger.info("Playout on : {0}".format(self.tree.get_path_to_node(node_id)))

        st_time = time.time()
        try:
            playout_node = self.env.rollout(self.tree.get_path_to_node(node_id))
        except Exception as e:
            # self.logger.error("Add node %s to not possible state: %s" % (node_id, e))
            self.tree.set_attribute(node_id, "invalid", True)
            return 0, None

        score = self.policy.evaluate(self.env._evaluate, [playout_node])

        self.logger.info(
            "param={0}\t score={1}\t exec time={2}".format(playout_node, score, time.time() - st_time))
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

    def run(self, nb_simulation=1, initial_configurations=[], step_to_generate_img=-1):
        """Run MCTS algorithm

        Parameters:
        ----------
        nb_simulation: int
            number of MCTS simulation to run (default is 10)
        initial_configurations: list of object
            set of configuration to start with (default is [])
        step_to_generate_img: int or None
            set of initial configuration (default -1, generate image for each MCTS iteration)
            Do not generate images if None.

        Returns:
        ----------
            int
                1 if timeout else 0

        """

        start_run = time.time()
        with Timeout(int(self.time_budget - (start_run - time.time()))):
            try:
                self.logger.info("Run default configuration")
                self.env.run_default_configuration()

                if len(initial_configurations) > 0:
                    self.logger.info("Run initial configurations")
                else:
                    self.logger.info("No initial configuration to run.")

                for i in range(nb_simulation):
                    if time.time() - self.env.start_time < self.time_budget:
                        res, config = self.MCT_SEARCH()

                        if res > self.best_score:
                            self.best_score = res
                            self.best_config = config
                    else:
                        return 0

                    if step_to_generate_img == -1 or i % step_to_generate_img == 0:
                        self.tree.draw_tree(
                            os.path.join(self.exec_dir, "images"))
                        self.print_tree("tree_{0}".format(i))

            except Timeout.Timeout:
                self.logger.info("Budget exhausted.")
                return 1

    def print_tree(self, name_img):
        """Print snapshot of constructed tree

        Parameters
        ----------
        name_img: str
            Path to store generated image
        """
        img_dir = os.path.join(self.exec_dir, "images")
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

        self.tree.draw_tree(os.path.join(img_dir, name_img))
