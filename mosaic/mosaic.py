import os
import sys
import logging
import tempfile

import numpy as np
from mosaic.mcts import MCTS


class Search:
    """
    Search optimal pipeline using Monte-Carlo Tree Search

    Parameters:
    ----------
        environment: object
            environment class extending AbstractEnvironment
        time_budget: int
            overall time budget
        seed: int
            random seed
        bandit_policy: dict
            bandit policy used in MCTS. Available choice are uct, besa, puct.
            Example {"policy_name": "uct", "c_ub": 1.41}, {"policy_name": "besa"}
        exec_dir: str
            directory to store tmp files

    Attributes
    ----------
    logger: class <logging>
        Logger used
    mcts : class <mosaic.MCTS>
        object that run MCTS algorithm

    """

    def __init__(self,
                 environment,
                 time_budget=3600,
                 verbose=False,
                 exec_dir=None,
                 bandit_policy=None,
                 seed=1,
                 coef_progressive_widening = 0.6):
        """Init method.
        """
        # config logger
        self.logger = logging.getLogger('mcts')
        self.logger.setLevel(logging.DEBUG)

        # Default bandit policy
        if bandit_policy is None:
            bandit_policy = {"policy_name": "uct", "c_uct": np.sqrt(2)}

        # execution directory
        if exec_dir is None:
            exec_dir = tempfile.mkdtemp()
        else:
            os.makedirs(exec_dir)

        hdlr = logging.FileHandler(os.path.join(exec_dir, "mcts.log"), mode='w')
        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(funcName)s :: %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        if verbose:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.mcts = MCTS(env=environment,
                         time_budget=time_budget,
                         exec_dir=exec_dir,
                         bandit_policy=bandit_policy,
                         coef_progressive_widening=coef_progressive_widening)

        np.random.seed(seed)

    def run(self, nb_simulation=10, initial_configurations=[], step_to_generate_img=-1):
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
            configuration: object
                best configuration

        """
        self.logger.info("# Run {0} iterations of MCTS".format(nb_simulation))
        self.mcts.run(nb_simulation, initial_configurations, step_to_generate_img)
        return self.mcts.best_config, self.mcts.best_score
