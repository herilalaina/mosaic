#TODO: Add comment


from mosaic.env import ConfigSpace_env
from mosaic.mcts import MCTS
import logging

class Search:
    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=360,
                 time_budget=3600,
                 multi_fidelity=False,
                 use_parameter_importance=False,
                 use_rave=False):
        env = ConfigSpace_env(eval_func,
                              config_space=config_space,
                              mem_in_mb=mem_in_mb,
                              cpu_time_in_s=cpu_time_in_s,
                              use_parameter_importance=use_parameter_importance,
                              use_rave=use_rave)
        self.mcts = MCTS(env = env,
                         time_budget=time_budget,
                         multi_fidelity=multi_fidelity)

        # config logger
        self.logger = logging.getLogger('mcts')
        hdlr = logging.FileHandler("mcts.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def print_config(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("logfile = {0}".format(self.logger))
        print("Use multi-fidelity = {0}".format(self.mcts.multi_fidelity))
        print("Use parameter importance = {0}".format(self.mcts.env.use_parameter_importance))
        print("Use RAVE = {0}".format(self.mcts.env.use_rave))
        print("Memory limit = {0} MB".format(self.mcts.env.mem_in_mb))
        print("Overall Time Budget = {0}".format(self.mcts.time_budget))
        print("Evaluation Time Limit = {0}".format(self.mcts.env.cpu_time_in_s))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def run(self, nb_simulation = 1, generate_image_path = ""):
        self.mcts.run(nb_simulation, generate_image_path)
        return self.mcts.env.bestconfig

    def run_warmstrat(self, eval_func,
                      mem_in_mb=3024,
                      cpu_time_in_s=360,
                      time_budget=3600,
                      nb_simulation = 1):
        self.mcts.env.reset(eval_func, mem_in_mb, cpu_time_in_s)
        self.mcts.reset(time_budget)
        self.run(nb_simulation=nb_simulation)

    def test_performance(self, X_train, y_train, X_test, y_test, func_test):
        scores = []
        for r in self.mcts.env.history_score:
            time = r["running_time"]
            model = r["model"]
            try:
                score = func_test(model, X_train, y_train, X_test, y_test)
                if score is not None:
                    scores.append((time, score))
            except Exception as e:
                print(e)
                pass
        return scores
