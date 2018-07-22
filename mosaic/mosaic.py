# TODO: Add comment


from mosaic.env import Env
from mosaic.mcts import MCTS
import logging


class Search:
    def __init__(self, scenario = None, sampler = {}, rules=[],
                 eval_func = None, logfile = '', widening_coef = 0.3):
        env = Env(scenario, sampler, rules, logfile)
        Env.evaluate = eval_func
        self.mcts = MCTS(env = env, widening_coef = widening_coef)
        Env.evaluate = eval_func

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
