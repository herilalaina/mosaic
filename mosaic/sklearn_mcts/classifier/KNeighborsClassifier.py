"""KNeighborsClassifier."""

import random
import numpy as np

from mosaic.sklearn_mcts.sklearn_space import Space_preprocessing
from mosaic.sklearn_mcts.sklearn_env import Env_preprocessing

from mosaic.space import Node_space
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Space_kneighborsClassifier(Space_preprocessing):
    def __init__(self):
        super(Space_kneighborsClassifier, self).__init__()

        kneighborsClassifier__n_neighbors = Node_space("kneighborsClassifier__n_neighbors")
        kneighborsClassifier__weights = kneighborsClassifier__n_neighbors.add_child("kneighborsClassifier__weights")

        self.append_to_parent(kneighborsClassifier__n_neighbors)

        # Add priors later
        self.default_params = {
            "algorithm": "auto",
            "n_jobs": -1
        }

        self.terminal_pointer = [kneighborsClassifier__weights]


class Env_kneighborsClassifier(Env_preprocessing):
    def __init__(self, X, y, aggreg_score=np.mean, ressource=None, cv=3, info=None):
        super(Env_kneighborsClassifier, self).__init__(X, y, aggreg_score)
        self.cv = cv
        self.info = info
        self.score_func = info["score_func"]

        sampler = self.space.sampler
        sample_to_add = {
            "kneighborsClassifier__n_neighbors": (random.choice, [range(1, 20)]),
            "kneighborsClassifier__weights": (random.choice, [["uniform", "distance"]])
        }
        sampler.update(sample_to_add)

        self.space = Space_kneighborsClassifier()
        self.space.set_sampler(sampler=sampler)

        self.list_estimator["kneighborsClassifier"] = KNeighborsClassifier
        self.terminal_state = ["kneighborsClassifier__weights"]

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

            estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            score = self.score_func(y_test, y_pred)

            if score < self.bestscore:
                for j in range(i, 3):
                    scores[j] = score
                return scores
            else:
                scores[i] = score

            i += 1
        return scores
