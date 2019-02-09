import math
import numpy as np
from mosaic.strategy import BaseStrategy, BaseEarlyStopping

class UCT(BaseStrategy, BaseEarlyStopping):
    def __init__(self):
        super().__init__()

    def selection(self, parent, ids, vals, visits, state=None):
        parent_val, parent_vis = parent
        return ids[np.argmax([(val + math.sqrt(2 * math.log10(parent_vis) / vis)) for vis, val in zip(visits, vals)])]

    def playout(self):
        pass

class PUCT(BaseStrategy, BaseEarlyStopping):
    def __init__(self, env, tree):
        super().__init__()
        self.env = env
        self.tree = tree

    def selection(self, parent, ids, vals, visits, state=None):
        perfs = []
        for id_child in ids:
            child = self.tree.get_info_node(id_child)
            perfs.append(self.env.estimate_action_state(state, child["name"], child["value"]))
        N = np.sum([np.exp(x) for x in perfs])
        probas = [np.exp(x) / N for x in perfs]
        return ids[np.argmax([val + 0.1 * prob * math.sqrt(sum(visits)) / (vis + 1) for vis, val, prob in zip(visits, vals, probas)])]

    def playout(self):
        pass

class Besa(UCT):
    def __init__(self):
        super(UCT, self).__init__()
        self.scores = dict()


    def selection(self, parent, ids, vals, visits, state=None):
        nb_count=min([len(self.scores[c]) for c in ids])
        if nb_count == 0:
            raise Exception("Need to check")
        else:
            new_val = []
            for id in ids:
                new_val.append(np.mean(np.random.choice(np.concatenate(list(self.scores.values())), size=nb_count)))
            return UCT.selection(self, parent, ids, new_val, [nb_count] * len(ids))

    def backpropagate(self, id, value, visit, reward):
        if id in self.scores:
            self.scores[id].append(reward)
        else:
            self.scores[id] = [reward]
        return UCT.backpropagate(self, id, value, visit, reward)
