import random
import warnings
import numpy as np

from mosaic.sklearn_mcts.sklearn_space import Space_preprocessing
from mosaic.sklearn_mcts.sklearn_env import Env_preprocessing

from mosaic.space import Node_space
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Space_logisticRegression(Space_preprocessing):
    def __init__(self):
        super(Space_logisticRegression, self).__init__()

        logisticRegression__penalty = Node_space("logisticRegression__penalty")
        self.append_to_parent(logisticRegression__penalty)

        logisticRegression__C = logisticRegression__penalty.add_child("logisticRegression__C")
        logisticRegression__max_iter = logisticRegression__C.add_child("logisticRegression__max_iter")

        self.default_params = {
            "solver": "saga",
            "class_weight": "balanced",
            "warm_start": True,
            "n_jobs": -1,
            "random_state": 42
        }

        self.terminal_pointer = [logisticRegression__max_iter]


class Env_logisticRegression(Env_preprocessing):
    def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
        super(Env_logisticRegression, self).__init__(X, y, aggreg_score)
        self.cv = cv

        reg_space = []
        for v in range(1, 10):
            reg_space.extend([v ** x for x in range(-10, 10)])

        sampler = self.space.sampler
        sample_to_add = {
            "logisticRegression__C": (random.choice, [reg_space]),
            "logisticRegression__penalty": (random.choice, [["l1", "l2"]]),
            "logisticRegression__max_iter": (random.choice, [[ressource]])
        }
        sampler.update(sample_to_add)

        self.space = Space_logisticRegression()
        self.space.set_sampler(sampler=sampler)

        self.early_stopping = 2
        self.list_estimator["logisticRegression"] = LogisticRegression
        self.terminal_state = ["logisticRegression__max_iter"]

    def set_ressource(self, val):
        self.space.sampler["logisticRegression__max_iter"] = (random.choice, [[val]])

    def calculate(self, moves):
        return self._calculate(moves)

    def compute_score(self, model, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=False, random_state=42)

        scores = [0] * self.cv
        i = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator = clone(model)

            with warnings.catch_warnings(record=True) as w:
                estimator.fit(X_train, y_train)

            y_pred = estimator.predict_proba(X_test)[:, 1]  # Get proba for y=1
            score = roc_auc_score(y_test, y_pred)

            if score < self.bestscore:
                for j in range(i, 3):
                    scores[j] = score
                return scores
            else:
                scores[i] = score

            i += 1
        return scores
