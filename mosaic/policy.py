"""Policy for bandit phase."""


import random
import math
import networkx as nx

class UCT():
    """Implements the UCT algorithm."""
    def __init__(self, use_rave = False):
        self.use_rave = use_rave

    def BESTCHILD(self, node, childs, scalar):
        """Return the best child."""
        bestscore = 0.0
        bestchildren = []
        for c in childs:
            score = self.uct(node, c, scalar)
            if score == bestscore:
                bestchildren.append(c["id"])
            if score > bestscore:
                bestchildren = [c["id"]]
                bestscore = score
        if len(bestchildren) == 0:
            raise Exception("No best child found")
        return random.choice(bestchildren)

    def uct(self, node, c, scalar):
        """Calculate value of node."""

        if float(c["visits"]) == 0:
            return 1000

        explore = math.sqrt(2.0 * math.log(node["visits"]) / float(c["visits"]))
        if not self.use_rave:
            score = c["reward"] + (scalar * explore)
            return score
        else:
            k = 200
            beta = math.sqrt(k / (3 * node["visits"] + k))
            rave_score = c["rave_score"]
            mc_score = c["reward"]
            uct_coef = (scalar * explore)
            return (1 - beta) * mc_score + beta * rave_score
