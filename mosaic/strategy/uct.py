import math
import numpy as np
from mosaic.strategy import BaseStrategy

class UCT(BaseStrategy):
    def __init__(self):
        super().__init__()

    def selection(self, parent, vals, visits):
        parent_val, parent_vis = parent
        return np.argmax([(val + math.sqrt(2 * math.log10(parent_vis) / vis)) for vis, val in zip(visits, vals)])

    def expansion(self):
        pass

    def playout(self):
        pass