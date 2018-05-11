from mosaic.mcts import MCTS

class Search(self):
    def __init__(self, env = None, logfile = ''):
        self.mcts = MCTS(env = env, logfile = logfile)

    def run(self, nb_simulation = 1, generate_image_path = ""):
        self.run(nb_simulation, generate_image_path)
        return self.mcts.env.bestconfig
