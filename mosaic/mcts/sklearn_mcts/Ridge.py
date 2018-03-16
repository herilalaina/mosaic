import random
import warnings
import numpy as np

from .sklearn_space import Space_preprocessing
from .sklearn_env import Env_preprocessing

from space import Node_space
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

class Space_ridge(Space_preprocessing):
	def __init__(self):
		super(Space_ridge, self).__init__()

		ridge__alpha = Node_space("ridge__alpha")
		ridge__max_iter = ridge__alpha.add_child("ridge__max_iter")
		self.append_to_parent(ridge__alpha)

		self.default_params = {
			"solver": "lsqr",
			"random_state": 42
		}

		self.terminal_pointer = [ridge__max_iter]


class Env_ridge(Env_preprocessing):
	def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
		super(Env_ridge, self).__init__(X, y, aggreg_score)
		self.cv = cv

		sampler = self.space.sampler
		alpha_space = []
		for v in range(1, 10):
			alpha_space.extend([ v**x for x in range(-10, 10) ])
		sample_to_add = {
			"ridge": (random.choice, [["ridge"]]),
			"ridge__alpha": (random.choice, [alpha_space]),
			"ridge__max_iter": (random.choice, [[ressource]]),
		}
		sampler.update(sample_to_add)

		self.space = Space_ridge()
		self.space.set_sampler(sampler = sampler)

		self.list_estimator["ridge"] = Ridge
		self.terminal_state = ["ridge__max_iter"]

	def set_ressource(self, val):
		self.space.sampler["ridge__max_iter"] = (random.choice, [[val]])

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
