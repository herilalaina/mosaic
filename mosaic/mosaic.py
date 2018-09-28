#TODO: Add comment


from mosaic.env import Env
from mosaic.mcts import MCTS
import logging


class Search:
    def __init__(self, scenario = None, sampler = {}, rules=[],
                 eval_func = None, logfile = ''):
        env = Env(eval_func, scenario, sampler, rules, logfile)
        self.mcts = MCTS(env = env)

        # config logger
        self.logger = logging.getLogger('mcts')
        hdlr = logging.FileHandler("mcts.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.DEBUG)

    def run(self, nb_simulation = 1, generate_image_path = ""):
        self.mcts.run(nb_simulation, generate_image_path)
        return self.mcts.env.bestconfig
