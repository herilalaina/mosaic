
import os
import logging
import tempfile
from mosaic.mcts import MCTS



class Search:
    """Main class to tune algorithm using Monte-Carlo Tree Search."""

    def __init__(self,
                 environment,
                 time_budget=3600,
                 seed=1,
                 policy_arg={},
                 exec_dir = None):
        """Initialization algorithm.

        :param environment: environment class extending AbstractEnvironment
        :param time_budget: overall time budget
        :param seed: random seed
        :param policy_arg: specific option for MCTS policy
        :param exec_dir: directory to store tmp files
        """
        env = environment
        # env.score_model.dataset_features = problem_features
        self.mcts = MCTS(env=env,
                         time_budget=time_budget,
                         policy_arg=policy_arg,
                         exec_dir=exec_dir)
        # config logger
        self.logger = logging.getLogger('mcts')

        # execution directory
        if exec_dir is not None:
            os.makedirs(exec_dir)
        else:
            exec_dir = tempfile.mkdtemp()

        log_dir = os.path.join(exec_dir, "mcts.log")
        hdlr = logging.FileHandler(log_dir, mode='w')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def run(self, nb_simulation = 1, initial_configurations=[], nb_iter_to_generate_img=-1):
        """Run MCTS algortihm

        :param nb_simulation: number of MCTS simulation to run
        :param initial_configurations: path for generated image , optional
        :param nb_iter_to_generate_img: set of initial configuration, optional
        :return:
        """
        self.logger.info("# Run {0} iterations of MCTS".format(nb_simulation))
        self.mcts.run(nb_simulation, initial_configurations, nb_iter_to_generate_img)
        return self.mcts.bestconfig, self.mcts.bestscore

    def test_performance(self, X_train, y_train, X_test, y_test, func_test, categorical_features):
        scores = []
        for r in self.mcts.env.final_model:
            time = r["running_time"]
            model = r["model"]
            try:
                score = func_test(model, X_train, y_train,
                                  X_test, y_test, categorical_features)
                if score is not None:
                    scores.append((time, score, r["cv_score"]))
            except Exception as e:
                self.logger.error(e)
        return scores
