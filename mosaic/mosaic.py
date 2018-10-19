#TODO: Add comment


from mosaic.env import ConfigSpace_env
from mosaic.mcts import MCTS
import logging


class Search:
    def __init__(self, eval_func,
                 config_space,
                 logfile = '',
                 multi_fidelity=False,
                 mem_in_mb=3024,
                 cpu_time_in_s=360,
                 time_budget=3600,
                 use_parameter_importance=True,
                 use_rave=False):
        env = ConfigSpace_env(eval_func,
                              config_space=config_space,
                              mem_in_mb=mem_in_mb,
                              cpu_time_in_s=cpu_time_in_s,
                              logfile = logfile,
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
