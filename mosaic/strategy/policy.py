import math
import time
import numpy as np
from mosaic.strategy import BaseStrategy, BaseEarlyStopping

class UCT(BaseStrategy, BaseEarlyStopping):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def selection(self, parent, ids, vals, visits, state=None):
        parent_val, parent_vis = parent
        return ids[np.argmax([(val + self.C * math.sqrt(math.log10(parent_vis) / vis)) for vis, val in zip(visits, vals)])]

    def playout(self):
        pass

class PUCT(BaseStrategy, BaseEarlyStopping):
    def __init__(self, env, tree, policy_arg):
        super().__init__()
        self.env = env
        self.tree = tree
        self.policy_arg = policy_arg
        self.scores = dict()

    def selection(self, parent, ids, vals, visits, state=None):
        perfs = []
        perfs_general = []
        for id_child in ids:
            child = self.tree.get_info_node(id_child)
            perfs.append(self.env.estimate_action_state(state, child["name"], child["value"]))
            perfs_general.append(self.env.estimate_action_state(state, child["name"], child["value"], local_model = False))
        N = np.sum([np.exp(x) for x in perfs])
        N_general = np.sum([np.exp(x) for x in perfs_general])
        probas = [np.exp(x) / N for x in perfs]
        probas_general = [np.exp(x) / N_general for x in perfs_general]

        beta = (time.time() - self.policy_arg["start_time"]) / self.policy_arg["time_budget"]

        # self.logger.info("## SELECTION ##")
        # self.logger.info("vals", vals)
        # self.logger.info("visits", visits)
        # self.logger.info("probas", probas)
        # self.logger.info("c=", self.policy_arg["c"])
        # self.logger.info("beta=", beta)
        res = [val + self.policy_arg["c"] * (prob) * math.sqrt(sum(visits)) / (vis + 1)
                for vis, val, prob, prob_gen in zip(visits, vals, probas, probas_general)]
        # self.logger.info("Final selection policy ", res)
        # self.logger.info("Selected ", np.argmax(res))

        return ids[np.argmax(res)]

    def playout(self):
        pass

    def backpropagate(self, id, value, visit, reward):
        if id in self.scores:
            self.scores[id].append(reward)
        else:
            self.scores[id] = [reward]
        return np.median(self.scores[id]), visit + 1

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
