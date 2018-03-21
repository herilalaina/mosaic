import random
import warnings
import numpy as np

from .sklearn_space import Space_sklearn
from .sklearn_env import Env_preprocessing

from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Space_lasso(Space_sklearn):
    def __init__(self):
        super(Space_lasso, self).__init__()

        lasso = self.root.add_child("lasso")
        lasso__alpha = lasso.add_child("lasso__alpha")
        lasso__max_iter = lasso__alpha.add_child("lasso__max_iter")

        self.default_params = {
            "selection": "random",
            "warm_start": True,
            "random_state": 42
        }

        self.terminal_pointer = [lasso__max_iter]


class Env_lasso(Env_preprocessing):
    def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
        super(Env_lasso, self).__init__(X, y, aggreg_score)
        self.cv = cv

        alpha_space = []
        for v in range(1, 10):
            alpha_space.extend([v ** x for x in range(-10, 10)])

        sample_to_add = {
            "lasso": (random.choice, [["lasso"]]),
            "lasso__alpha": (random.choice, [alpha_space]),
            "lasso__max_iter": (random.choice, [[ressource]]),
        }

        self.space = Space_lasso()
        self.space.set_sampler(sampler=sample_to_add)

        self.list_estimator["lasso"] = Lasso
        self.terminal_state = ["lasso__max_iter"]
        self.have_preprocessing = False

    def set_ressource(self, val):
        self.space.sampler["lasso__max_iter"] = (random.choice, [[val]])

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

            y_pred = estimator.predict(X_test)
            y_pred[y_pred > 1] = 1
            y_pred[y_pred < 0] = 0

            scores[i] = roc_auc_score(y_test, y_pred)

            i += 1
        return scores
