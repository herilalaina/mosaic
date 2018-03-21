import random
import warnings
import numpy as np

from .sklearn_space import Space_preprocessing
from .sklearn_env import Env_preprocessing

from mosaic.space import Node_space
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

class Space_xgboost(Space_preprocessing):
	def __init__(self):
		super(Space_xgboost, self).__init__()

		xgboost__learning_rate = Node_space("xgboost__learning_rate")
		xgboost__max_depth = xgboost__learning_rate.add_child("xgboost__max_depth")
		xgboost__gamma = xgboost__max_depth.add_child("xgboost__gamma")
		xgboost__subsample = xgboost__gamma.add_child("xgboost__subsample")
		xgboost__reg_alpha = xgboost__subsample.add_child("xgboost__reg_alpha")
		xgboost__reg_lambda = xgboost__reg_alpha.add_child("xgboost__reg_lambda")
		xgboost__n_estimators = xgboost__reg_lambda.add_child("xgboost__n_estimators")

		self.append_to_parent(xgboost__learning_rate)

		# Add priors later
		self.default_params = {
			"n_jobs": -1,
			"random_state": 42
		}

		self.terminal_pointer = [xgboost__n_estimators]


class Env_xgboost(Env_preprocessing):
	def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
		super(Env_xgboost, self).__init__(X, y, aggreg_score=np.min)
		self.cv =cv

		sampler = self.space.sampler
		sample_to_add = {
			"xgboost__learning_rate": (random.uniform, [0.01, 0.2]),
			"xgboost__max_depth": (random.randint, [1, 50]),
			"xgboost__gamma": (random.uniform, [0.0, 0.001]),
			"xgboost__subsample": (random.uniform, [0.1, 0.9]),
			"xgboost__reg_alpha": (random.uniform, [0, 1]),
			"xgboost__reg_lambda": (random.uniform, [0, 1]),
			"xgboost__n_estimators": (random.choice, [[ressource]])
		}
		sampler.update(sample_to_add)

		self.space = Space_xgboost()
		weight = compute_class_weight("balanced", [0, 1], self.y)
		self.space.default_params["scale_pos_weight"] = weight[0]/weight[1]
		self.space.set_sampler(sampler=sampler)

		self.list_estimator["xgboost"] = XGBClassifier
		self.terminal_state = ["xgboost__n_estimators"]

	def set_ressource(self, val):
		self.space.sampler["xgboost__n_estimators"] = (random.choice, [[val]])

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

			estimator.fit(X_train, y_train)

			y_pred = estimator.predict_proba(X_test)[:, 1] # Get proba for y=1
			score = roc_auc_score(y_test, y_pred)

			if score < self.bestscore:
				for j in range(i, 3):
					scores[j] = score
				return scores
			else:
				scores[i] = score

			i += 1
		return scores
