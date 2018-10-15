#TODO: Add comment


from mosaic.env import ConfigSpace_env
from mosaic.mcts import MCTS
import logging


class Search:
    def __init__(self, eval_func, config_space, logfile = '', mem_in_mb=3024, cpu_time_in_s=360):
        env = ConfigSpace_env(eval_func, config_space=config_space, mem_in_mb=mem_in_mb, cpu_time_in_s=360, logfile = logfile)
        self.mcts = MCTS(env = env)

        # config logger
        self.logger = logging.getLogger('mcts')
        hdlr = logging.FileHandler("mcts.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        #self.logger.setLevel(logging.DEBUG)

    def run(self, nb_simulation = 1, generate_image_path = ""):
        self.mcts.run(nb_simulation, generate_image_path)
        return self.mcts.env.bestconfig
