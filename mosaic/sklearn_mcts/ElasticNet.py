import random
import warnings
import numpy as np

from .sklearn_space import Space_sklearn
from .sklearn_env import Env_preprocessing

from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Space_elasticNet(Space_sklearn):
    def __init__(self):
        super(Space_elasticNet, self).__init__()

        elasticNet = self.root.add_child("elasticNet")
        elasticNet__alpha = elasticNet.add_child("elasticNet__alpha")
        elasticNet__positive = elasticNet__alpha.add_child("elasticNet__positive")
        elasticNet__l1_ratio = elasticNet__positive.add_child("elasticNet__l1_ratio")
        elasticNet__max_iter = elasticNet__l1_ratio.add_child("elasticNet__max_iter")

        self.default_params = {
            "selection": "random",
            "warm_start": True,
            "random_state": 42
        }

        self.terminal_pointer = [elasticNet__max_iter]


class Env_elasticNet(Env_preprocessing):
    def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
        super(Env_elasticNet, self).__init__(X, y, aggreg_score)

        self.cv = cv

        alpha_space = []
        for v in range(1, 10):
            alpha_space.extend([v ** x for x in range(-10, 10)])
        sample_to_add = {
            "elasticNet": (random.choice, [["elasticNet"]]),
            "elasticNet__alpha": (random.choice, [alpha_space]),
            "elasticNet__l1_ratio": (random.uniform, [0, 1]),
            "elasticNet__positive": (random.choice, [[True, False]]),
            "elasticNet__max_iter": (random.choice, [[ressource]])
        }

        self.space = Space_elasticNet()
        self.space.set_sampler(sampler=sample_to_add)

        self.list_estimator["elasticNet"] = ElasticNet
        self.terminal_state = ["elasticNet__max_iter"]
        self.have_preprocessing = False

    def set_ressource(self, val):
        self.space.sampler["elasticNet__max_iter"] = (random.choice, [[val]])

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

            repeat = True
            with warnings.catch_warnings(record=True) as w:
                estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            y_pred[y_pred > 1] = 1
            y_pred[y_pred < 0] = 0

            scores[i] = roc_auc_score(y_test, y_pred)

            i += 1
        return scores
