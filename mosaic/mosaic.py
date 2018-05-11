from mosaic.env import Env
from mosaic.mcts import MCTS

class Search():
    def __init__(self, scenario = None, sampler = {}, rules = [], eval_func = None, logfile = ''):
        env = Env(scenario, sampler, rules)
        Env.evaluate = eval_func
        self.mcts = MCTS(env = env, logfile = logfile)
        Env.evaluate = eval_func

    def run(self, nb_simulation = 1, generate_image_path = ""):
        self.mcts.run(nb_simulation, generate_image_path)
        return self.mcts.env.bestconfig
