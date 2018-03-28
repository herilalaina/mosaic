import random
import warnings
import numpy as np

from mosaic.sklearn_mcts.sklearn_space import Space_preprocessing
from mosaic.sklearn_mcts.sklearn_env import Env_preprocessing

from mosaic.space import Node_space
from mosaic import sklearn_mcts
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Space_linearDiscriminantAnalysis(Space_preprocessing):
    def __init__(self):
        super(Space_linearDiscriminantAnalysis, self).__init__()

        linearDiscriminantAnalysis__shrinkage = Node_space("linearDiscriminantAnalysis__shrinkage")
        self.append_to_parent(linearDiscriminantAnalysis__shrinkage)

        # Add priors later
        self.default_params = {
            "solver": "lsqr"
        }

        self.terminal_pointer = [linearDiscriminantAnalysis__shrinkage]


class Env_linearDiscriminantAnalysis(Env_preprocessing):
    def __init__(self, X, y, aggreg_score=np.mean, ressource=None, cv=3, info=None):
        super(Env_linearDiscriminantAnalysis, self).__init__(X, y, aggreg_score)

        self.cv = cv
        self.info = info
        self.score_func = info["score_func"]

        sampler = self.space.sampler
        sample_to_add = {
            "linearDiscriminantAnalysis__shrinkage": (random.uniform, [0, 1])
        }
        sampler.update(sample_to_add)

        self.space = Space_linearDiscriminantAnalysis()
        self.space.set_sampler(sampler=sampler)
        ratio_1 = float(len(self.y[self.y == 1]) / len(self.y))
        self.space.default_params["priors"] = [1 - ratio_1, ratio_1]

        self.early_stopping = 2
        self.list_estimator["linearDiscriminantAnalysis"] = LinearDiscriminantAnalysis
        self.terminal_state = ["linearDiscriminantAnalysis__shrinkage"]

    def set_ressource(self, val):
        pass

    def calculate(self, moves):
        return self._calculate(moves)

    def compute_score(self, model, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=False)

        scores = [0] * self.cv
        i = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator = clone(model)

            warnings.filterwarnings("ignore")
            estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            score = sklearn_mcts.calculate_score_metric(estimator, X_test, y_test, self.info)

            if score < self.bestscore:
                for j in range(i, 3):
                    scores[j] = score
                return scores
            else:
                scores[i] = score

            i += 1
        return scores
