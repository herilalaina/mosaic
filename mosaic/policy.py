"""Policy for bandit phase."""


import random
import math
import networkx as nx

class UCT():
    """Implements the UCT algorithm."""

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
        exploit = c["reward"] / c["visits"]
        score = exploit + (scalar * explore)
        return score
