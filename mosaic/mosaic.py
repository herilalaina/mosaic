
import os
import logging
from mosaic.mcts import MCTS


class Search:
    """Main class to tune algorithm using Monte-Carlo Tree Search."""

    def __init__(self,
                 environment,
                 time_budget=3600,
                 seed=1,
                 policy_arg={},
                 exec_dir=""):
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
        if exec_dir:
            os.makedirs(exec_dir)
            log_dir = os.path.join(exec_dir, "mcts.log")
        else:
            log_dir = "mcts.log"

        hdlr = logging.handlers.RotatingFileHandler(log_dir, maxBytes=1024,backupCount=1)
        #logging.FileHandler(log_dir, mode='w')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def print_config(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("logfile = {0}".format(self.logger))
        print(self.mcts.env)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def run(self, nb_simulation=1, generate_image_path="", intial_configuration=[]):
        """Run MCTS algortihm

        :param nb_simulation: number of MCTS simulation to run
        :param generate_image_path: path for generated image , optional
        :param intial_configuration: set of initial configuration, optional
        :return:
        """
        self.print_config()
        self.mcts.run(nb_simulation, intial_configuration, generate_image_path)
        return self.mcts.bestconfig, self.mcts.bestscore

    def get_history_run(self):
        return self.mcts.env.final_model

    def get_full_history(self):
        return self.mcts.env.final_model

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
                print(e)
                pass
        return scores
