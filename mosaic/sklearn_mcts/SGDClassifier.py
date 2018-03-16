import random
import warnings
import numpy as np

from .sklearn_space import Space_preprocessing
from .sklearn_env import Env_preprocessing

from space import Node_space
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

class Space_sgdClassifier(Space_preprocessing):
	def __init__(self):
		super(Space_sgdClassifier, self).__init__()

		sgdClassifier__loss = Node_space("sgdClassifier__loss")
		sgdClassifier__learning_rate = sgdClassifier__loss.add_child("sgdClassifier__learning_rate")
		sgdClassifier__penalty = sgdClassifier__learning_rate.add_child("sgdClassifier__penalty")
		sgdClassifier__alpha = sgdClassifier__penalty.add_child("sgdClassifier__alpha")
		sgdClassifier__l1_ratio = sgdClassifier__alpha.add_child("sgdClassifier__l1_ratio")
		sgdClassifier__max_iter = sgdClassifier__l1_ratio.add_child("sgdClassifier__max_iter")

		self.append_to_parent(sgdClassifier__loss)

		self.default_params = {
			"n_jobs": 2,
			"class_weight": "balanced",
			"warm_start": True,
			"eta0": 0.1,
			"random_state": 42
		}

		self.terminal_pointer = [sgdClassifier__max_iter]


class Env_sgdClassifier(Env_preprocessing):
	def __init__(self, X, y, aggreg_score=np.mean, ressource=1, cv=3):
		super(Env_sgdClassifier, self).__init__(X, y, aggreg_score)
		self.cv = cv

		sampler = self.space.sampler
		alpha_space = []
		for v in range(1, 10):
			alpha_space.extend([ v**x for x in range(-10, 10) ])
		sample_to_add = {
			"sgdClassifier__loss": (random.choice, [["hinge", "modified_huber"]]),
			"sgdClassifier__penalty": (random.choice, [["l1", "l2"]]),
			"sgdClassifier__alpha": (random.choice, [alpha_space]),
			"sgdClassifier__l1_ratio": (random.uniform, [0, 1]),
			"sgdClassifier__learning_rate": (random.choice, [["optimal", "invscaling"]]),
			"sgdClassifier__max_iter": (random.choice, [[ressource]]),
		}
		sampler.update(sample_to_add)

		self.space = Space_sgdClassifier()
		self.space.set_sampler(sampler=sampler)

		self.early_stopping = 2
		self.list_estimator["sgdClassifier"] = SGDClassifier
		self.terminal_state = ["sgdClassifier__max_iter"]

	def set_ressource(self, val):
		self.space.sampler["sgdClassifier__max_iter"] = (random.choice, [[val]])

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

			#with warnings.catch_warnings(record=True) as w:
			estimator.fit(X_train, y_train)

			# Get proba for y=1
			try:
				y_pred = estimator.predict_proba(X_test)[:, 1]
			except:
				y_pred = estimator.predict(X_test)
			score = roc_auc_score(y_test, y_pred)

			if score < self.bestscore:
				for j in range(i, 3):
					scores[j] = score
				return scores
			else:
				scores[i] = score

			i += 1
		return scores

	def parse_params(self, moves):
		list_estimator_params = {}
		pipeline = []

		for name, value in moves[1:]:
			l_name = name.split("__")
			if len(l_name) == 1 or pipeline[-1][0] != l_name[0]:
				name_estimator = l_name[0]
				if name_estimator == "sgdClassifier":
					pipeline.append([name_estimator, self.get_estimator(name_estimator)(**self.space.default_params)])
				else:
					pipeline.append([name_estimator, self.get_estimator(name_estimator)()])
			else:
				name_estimator, estimator_param = l_name[0], l_name[1]

				params = {estimator_param : self.get_estimator(value)}
				pipeline[-1][1].set_params(**params)

		return [(a[0],a[1]) for a in pipeline]
